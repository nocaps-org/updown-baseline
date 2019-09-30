import argparse
import os
from typing import Any, Dict, List

import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from allennlp.data import Vocabulary

from updown.config import Config
from updown.data.datasets import (
    TrainingDataset,
    EvaluationDataset,
    EvaluationDatasetWithConstraints,
)
from updown.models import UpDownCaptioner
from updown.types import Prediction
from updown.utils.checkpointing import CheckpointManager
from updown.utils.common import cycle
from updown.utils.evalai import NocapsEvaluator
from updown.utils.constraints import add_constraint_words_to_vocabulary


parser = argparse.ArgumentParser("Train an UpDown Captioner on COCO train2017 split.")
parser.add_argument(
    "--config", required=True, help="Path to a config file with all configuration parameters."
)
parser.add_argument(
    "--config-override",
    default=[],
    nargs="*",
    help="A sequence of key-value pairs specifying certain config arguments (with dict-like "
    "nesting) using a dot operator. The actual config will be updated and recorded in "
    "the serialization directory.",
)

parser.add_argument_group("Compute resource management arguments.")
parser.add_argument(
    "--gpu-ids", required=True, nargs="+", type=int, help="List of GPU IDs to use (-1 for CPU)."
)
parser.add_argument(
    "--cpu-workers", type=int, default=0, help="Number of CPU workers to use for data loading."
)
parser.add_argument(
    "--in-memory", action="store_true", help="Whether to load image features in memory."
)

parser.add_argument_group("Checkpointing related arguments.")
parser.add_argument(
    "--skip-validation",
    action="store_true",
    help="Whether to skip validation and simply serialize checkpoints. This won't track the "
    "best performing checkpoint (obviously). useful for cases where GPU server does not have "
    "internet access and/or checkpoints are validation externally.",
)
parser.add_argument(
    "--serialization-dir",
    default="checkpoints/experiment",
    help="Path to a (non-existent) directory for serializing checkpoints and tensorboard logs.",
)
parser.add_argument(
    "--checkpoint-every",
    default=1000,
    type=int,
    help="Save a checkpoint after every this many epochs/iterations.",
)
parser.add_argument(
    "--start-from-checkpoint",
    default="",
    help="Path to load checkpoint and continue training [only supported for module_training].",
)


if __name__ == "__main__":
    # --------------------------------------------------------------------------------------------
    #   INPUT ARGUMENTS AND CONFIG
    # --------------------------------------------------------------------------------------------
    _A = parser.parse_args()

    # Create a config with default values, then override from config file, and _A.
    # This config object is immutable, nothing can be changed in this anymore.
    _C = Config(_A.config, _A.config_override)

    # Print configs and args.
    print(_C)
    for arg in vars(_A):
        print("{:<20}: {}".format(arg, getattr(_A, arg)))

    # Create serialization directory and save config in it.
    os.makedirs(_A.serialization_dir, exist_ok=True)
    _C.dump(os.path.join(_A.serialization_dir, "config.yml"))

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    # These five lines control all the major sources of randomness.
    np.random.seed(_C.RANDOM_SEED)
    torch.manual_seed(_C.RANDOM_SEED)
    torch.cuda.manual_seed_all(_C.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set device according to specified GPU ids.
    device = torch.device(f"cuda:{_A.gpu_ids[0]}" if _A.gpu_ids[0] >= 0 else "cpu")

    # --------------------------------------------------------------------------------------------
    #   INSTANTIATE VOCABULARY, DATALOADER, MODEL, OPTIMIZER
    # --------------------------------------------------------------------------------------------

    vocabulary = Vocabulary.from_files(_C.DATA.VOCABULARY)

    # If we wish to use CBS during evaluation or inference, expand the vocabulary and add
    # constraint words derived from Open Images classes.
    if _C.MODEL.USE_CBS:
        vocabulary = add_constraint_words_to_vocabulary(
            vocabulary, wordforms_tsvpath=_C.DATA.CBS.WORDFORMS
        )

    train_dataset = TrainingDataset.from_config(_C, vocabulary=vocabulary, in_memory=_A.in_memory)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE,
        shuffle=True,
        num_workers=_A.cpu_workers,
        collate_fn=train_dataset.collate_fn,
    )
    # Make dataloader cyclic for sampling batches perpetually.
    train_dataloader = cycle(train_dataloader, device)

    EvaluationDatasetClass = (
        EvaluationDatasetWithConstraints if _C.MODEL.USE_CBS else EvaluationDataset
    )
    val_dataset = EvaluationDatasetClass.from_config(
        _C, vocabulary=vocabulary, in_memory=_A.in_memory
    )

    # Use a smaller batch during validation (accounting beam size) to fit in memory.
    val_batch_size = _C.OPTIM.BATCH_SIZE // _C.MODEL.BEAM_SIZE

    # Reduce batch size by total FSM states during CBS, because net beam size is larger.
    if _C.MODEL.USE_CBS:
        val_batch_size = val_batch_size // (2 ** _C.DATA.CBS.MAX_GIVEN_CONSTRAINTS)
        val_batch_size = val_batch_size or 1

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=_A.cpu_workers,
        collate_fn=val_dataset.collate_fn,
    )

    model = UpDownCaptioner.from_config(_C, vocabulary=vocabulary).to(device)
    if len(_A.gpu_ids) > 1 and -1 not in _A.gpu_ids:
        # Don't wrap to DataParallel if single GPU ID or -1 (CPU) is provided.
        model = nn.DataParallel(model, _A.gpu_ids)

    optimizer = optim.SGD(
        model.parameters(),
        lr=_C.OPTIM.LR,
        momentum=_C.OPTIM.MOMENTUM,
        weight_decay=_C.OPTIM.WEIGHT_DECAY,
    )
    lr_scheduler = optim.lr_scheduler.LambdaLR(  # type: ignore
        optimizer, lr_lambda=lambda iteration: 1 - iteration / _C.OPTIM.NUM_ITERATIONS
    )

    # --------------------------------------------------------------------------------------------
    #  BEFORE TRAINING STARTS
    # --------------------------------------------------------------------------------------------

    # Tensorboard summary writer for logging losses and metrics.
    tensorboard_writer = SummaryWriter(logdir=_A.serialization_dir)

    # Checkpoint manager to serialize checkpoints periodically while training and keep track of
    # best performing checkpoint.
    checkpoint_manager = CheckpointManager(model, optimizer, _A.serialization_dir, mode="max")

    # Evaluator submits predictions to EvalAI and retrieves results.
    evaluator = NocapsEvaluator(phase="val")

    # Load checkpoint to resume training from there if specified.
    # Infer iteration number through file name (it's hacky but very simple), so don't rename
    # saved checkpoints if you intend to continue training.
    if _A.start_from_checkpoint != "":
        training_checkpoint: Dict[str, Any] = torch.load(_A.start_from_checkpoint)
        for key in training_checkpoint:
            if key == "optimizer":
                optimizer.load_state_dict(training_checkpoint[key])
            else:
                # Don't complain about missing embeddings, they might be absent if frozen.
                model.load_state_dict(training_checkpoint[key])
        start_iteration = int(_A.start_from_checkpoint.split("_")[-1][:-4]) + 1
    else:
        start_iteration = 1

    # --------------------------------------------------------------------------------------------
    #   TRAINING LOOP
    # --------------------------------------------------------------------------------------------
    for iteration in tqdm(range(start_iteration, _C.OPTIM.NUM_ITERATIONS + 1)):

        # keys: {"image_id", "image_features", "caption_tokens"}
        batch = next(train_dataloader)

        optimizer.zero_grad()
        output_dict = model(batch["image_features"], batch["caption_tokens"])
        batch_loss = output_dict["loss"].mean()

        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), _C.OPTIM.CLIP_GRADIENTS)

        optimizer.step()
        lr_scheduler.step()

        # Log loss and learning rate to tensorboard.
        tensorboard_writer.add_scalar("loss", batch_loss, iteration)
        tensorboard_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], iteration)

        # ----------------------------------------------------------------------------------------
        #   VALIDATION
        # ----------------------------------------------------------------------------------------
        if iteration % _A.checkpoint_every == 0:
            if not _A.skip_validation:
                model.eval()
                predictions: List[Prediction] = []

                for batch in tqdm(val_dataloader):
                    # keys: {"image_id", "image_features"}
                    batch = {key: value.to(device) for key, value in batch.items()}

                    with torch.no_grad():
                        # shape: (batch_size, max_caption_length)
                        # Pass finite state machine and number of constraints if using CBS.
                        batch_predictions = model(
                            batch["image_features"],
                            fsm=batch.get("fsm", None),
                            num_constraints=batch.get("num_constraints", None),
                        )["predictions"]

                    for i, image_id in enumerate(batch["image_id"]):
                        instance_predictions = batch_predictions[i, :]

                        # De-tokenize caption tokens and trim until first "@@BOUNDARY@@".
                        caption = [
                            vocabulary.get_token_from_index(p.item()) for p in instance_predictions
                        ]
                        eos_occurences = [
                            j for j in range(len(caption)) if caption[j] == "@@BOUNDARY@@"
                        ]
                        caption = (
                            caption[: eos_occurences[0]] if len(eos_occurences) > 0 else caption
                        )
                        predictions.append(
                            {"image_id": image_id.item(), "caption": " ".join(caption)}
                        )

                model.train()

                # Print first 25 captions with their Image ID.
                for k in range(25):
                    print(predictions[k]["image_id"], predictions[k]["caption"])

                # Get evaluation metrics for nocaps val phase from EvalAI.
                # keys: {"B1", "B2", "B3", "B4", "METEOR", "ROUGE-L", "CIDEr", "SPICE"}
                # In each of these, keys:  {"in-domain", "near-domain", "out-domain", "entire"}
                evaluation_metrics = evaluator.evaluate(predictions, iteration)

                # Print and log all evaluation metrics to tensorboard.
                print(f"Evaluation metrics after iteration {iteration}:")
                for metric_name in evaluation_metrics:
                    tensorboard_writer.add_scalars(
                        f"metrics/{metric_name}", evaluation_metrics[metric_name], iteration
                    )
                    print(f"\t{metric_name}:")
                    for domain in evaluation_metrics[metric_name]:
                        print(f"\t\t{domain}:", evaluation_metrics[metric_name][domain])

                # Serialize checkpoint and update best checkpoint by overall CIDEr.
                checkpoint_manager.step(evaluation_metrics["CIDEr"]["entire"], iteration)
            else:
                # Serialize checkpoint, best checkpoint does not mean anything if skipping
                # validation altogether.
                checkpoint_manager.step(0.0, iteration)
