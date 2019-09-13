import argparse
import json
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from allennlp.data import Vocabulary

from updown.config import Config
from updown.data.datasets import EvaluationDataset, EvaluationDatasetWithConstraints
from updown.models import UpDownCaptioner
from updown.types import Prediction
from updown.utils.evalai import NocapsEvaluator
import updown.utils.cbs as cbs_utils


parser = argparse.ArgumentParser(
    "Run inference using UpDown Captioner, on either nocaps val or test split."
)
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
parser.add_argument("--run-val", action="store_true", help="Whether to run val data")
parser.add_argument(
    "--checkpoint-path", required=True, help="Path to load checkpoint and run inference on."
)
parser.add_argument("--output-path", required=True, help="Path to save predictions (as a JSON).")
parser.add_argument(
    "--evalai-submit", action="store_true", help="Whether to submit the predictions to EvalAI."
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
    #   INSTANTIATE VOCABULARY, DATALOADER, MODEL
    # --------------------------------------------------------------------------------------------

    vocabulary = Vocabulary.from_files(_C.DATA.VOCABULARY)

    # If we wish to use CBS during evaluation or inference, expand the vocabulary and add
    # constraint words derived from Open Images classes.
    if _C.MODEL.USE_CBS:
        vocabulary = cbs_utils.add_constraint_words_to_vocabulary(
            vocabulary, constraint_words_filepath=_C.DATA.CBS_OPEN_IMAGE_WORD_FORM
        )

    if _C.MODEL.USE_CBS:
        eval_dataset = EvaluationDatasetWithConstraints(
            vocabulary,
            image_features_h5path=_C.DATA.TEST_FEATURES
            if not _A.run_val
            else _C.DATA.VAL_FEATURES,
            boxes_jsonpath=_C.DATA.CBS_TEST_CONSTRAINTS
            if not _A.run_val
            else _C.DATA.CBS_VAL_CONSTRAINTS,
            constraint_wordforms_csvpath=_C.DATA.CBS_OPEN_IMAGE_WORD_FORM,
            hierarchy_jsonpath=_C.DATA.CBS_CLASS_HIERARCHY_PATH,
            in_memory=_A.in_memory,
        )
    else:
        eval_dataset = EvaluationDataset(
            image_features_h5path=_C.DATA.TEST_FEATURES
            if not _A.run_val
            else _C.DATA.VAL_FEATURES,
            in_memory=_A.in_memory,
        )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE // _C.MODEL.BEAM_SIZE,
        shuffle=False,
        num_workers=_A.cpu_workers,
        collate_fn=eval_dataset.collate_fn,
    )

    model = UpDownCaptioner(
        vocabulary,
        image_feature_size=_C.MODEL.IMAGE_FEATURE_SIZE,
        embedding_size=_C.MODEL.EMBEDDING_SIZE,
        hidden_size=_C.MODEL.HIDDEN_SIZE,
        attention_projection_size=_C.MODEL.ATTENTION_PROJECTION_SIZE,
        beam_size=_C.MODEL.BEAM_SIZE,
        max_caption_length=_C.DATA.MAX_CAPTION_LENGTH,
    ).to(device)

    # Load checkpoint to run inference.
    model.load_state_dict(torch.load(_A.checkpoint_path)["model"])

    if len(_A.gpu_ids) > 1 and -1 not in _A.gpu_ids:
        # Don't wrap to DataParallel if single GPU ID or -1 (CPU) is provided.
        model = nn.DataParallel(model, _A.gpu_ids)

    # --------------------------------------------------------------------------------------------
    #   INFERENCE LOOP
    # --------------------------------------------------------------------------------------------
    model.eval()

    predictions: List[Prediction] = []

    for batch in tqdm(eval_dataloader):

        # keys: {"image_id", "image_features"}
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.no_grad():
            # shape: (batch_size, max_caption_length)
            batch_predictions = model(
                batch["image_id"],
                batch["image_features"],
                state_transition_matrix=batch.get("state_transition_matrix", None),
                num_candidates=batch.get("num_candidates", None),
            )["predictions"]

        for i, image_id in enumerate(batch["image_id"]):
            instance_predictions = batch_predictions[i, :]

            # De-tokenize caption tokens and trim until first "@@BOUNDARY@@".
            caption = [vocabulary.get_token_from_index(p.item()) for p in instance_predictions]
            eos_occurences = [j for j in range(len(caption)) if caption[j] == "@@BOUNDARY@@"]
            caption = caption[: eos_occurences[0]] if len(eos_occurences) > 0 else caption

            predictions.append({"image_id": image_id.item(), "caption": " ".join(caption)})

    # Print first 25 captions with their Image ID.
    for k in range(25):
        print(predictions[k]["image_id"], predictions[k]["caption"])

    json.dump(predictions, open(_A.output_path, "w"))

    if _A.evalai_submit:
        evaluator = NocapsEvaluator("val" if _A.run_val else "test")
        evaluation_metrics = evaluator.evaluate(predictions)

        print(f"Evaluation metrics for checkpoint {_A.checkpoint_path}:")
        for metric_name in evaluation_metrics:
            print(f"\t{metric_name}:")
            for domain in evaluation_metrics[metric_name]:
                print(f"\t\t{domain}:", evaluation_metrics[metric_name][domain])
