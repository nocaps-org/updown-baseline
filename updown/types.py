from typing import List

from mypy_extensions import TypedDict
import numpy as np
import torch


# Type hint for objects returned by ``TrainingDataset.__getitem__``.
TrainingInstance = TypedDict(
    "TrainingInstance",
    {"image_id": int, "image_features": np.ndarray, "caption_tokens": List[int]},
)

# Type hint for objects returned by ``TrainingDataset.collate_fn``.
TrainingBatch = TypedDict(
    "TrainingBatch",
    {
        "image_id": torch.LongTensor,
        "image_features": torch.FloatTensor,
        "caption_tokens": torch.LongTensor,
    },
)

# Type hint for objects returned by ``EvaluationDataset.__getitem__``.
EvaluationInstance = TypedDict(
    "EvaluationInstance", {"image_id": int, "image_features": np.ndarray}
)
EvaluationInstanceWithConstraints = TypedDict(
    "EvaluationInstanceWithConstraints",
    {
        "image_id": int,
        "image_features": np.ndarray,
        "fsm": torch.ByteTensor,
        "num_states": int,
        "num_constraints": int,
    },
)

# Type hint for objects returned by ``EvaluationDataset.collate_fn``.
EvaluationBatch = TypedDict(
    "EvaluationBatch", {"image_id": torch.LongTensor, "image_features": torch.FloatTensor}
)
EvaluationBatchWithConstraints = TypedDict(
    "EvaluationBatchWithConstraints",
    {
        "image_id": int,
        "image_features": torch.FloatTensor,
        "fsm": torch.ByteTensor,
        "num_constraints": torch.LongTensor,
    },
)

ConstraintBoxes = TypedDict(
    "ConstraintBoxes", {"boxes": np.ndarray, "class_names": List[str], "score": np.ndarray}
)

Prediction = TypedDict("Prediction", {"image_id": int, "caption": str})
