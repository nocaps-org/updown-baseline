from typing import List

import numpy as np

import torch
from torch.utils.data import Dataset
from allennlp.data import Vocabulary

from updown.data.readers import CocoCaptionsReader, ImageFeaturesReader
from updown.types import TrainingInstance, TrainingBatch, ValidationInstance, ValidationBatch


def _collate_image_features(image_features_list: List[np.ndarray]) -> np.ndarray:
    # This will be (num_boxes * image_features_size) for adaptive features because of the way
    # these features are saved in H5 file.
    image_features_dim = [instance.shape[0] for instance in image_features_list]

    image_features = np.zeros(
        (len(image_features_list), max(image_features_dim)), dtype=np.float32
    )
    for i, (instance, dim) in enumerate(zip(image_features_list, image_features_dim)):
        image_features[i, :dim] = instance
    return image_features


class TrainingDataset(Dataset):
    def __init__(
        self,
        vocabulary: Vocabulary,
        captions_jsonpath: str,
        image_features_h5path: str,
        max_caption_length: int = 20,
        in_memory: bool = True,
    ) -> None:

        self._vocabulary = vocabulary
        self._image_features_reader = ImageFeaturesReader(image_features_h5path, in_memory)
        self._captions_reader = CocoCaptionsReader(captions_jsonpath)

        self._max_caption_length = max_caption_length

    def __len__(self) -> int:
        # Number of training examples are number of captions, not number of images.
        return len(self._captions_reader)

    def __getitem__(self, index: int) -> TrainingInstance:

        image_id, caption = self._captions_reader[index]
        image_features = self._image_features_reader[image_id]

        # Tokenize caption.
        caption_tokens: List[int] = [self._vocabulary.get_token_index(c) for c in caption]

        # Pad upto max_caption_length.
        caption_tokens = caption_tokens[: self._max_caption_length]
        caption_tokens.extend(
            [self._vocabulary.get_token_index("@@UNKNOWN@@")]
            * (self._max_caption_length - len(caption_tokens))
        )

        item: TrainingInstance = {
            "image_id": image_id,
            "image_features": image_features,
            "caption_tokens": caption_tokens,
        }
        return item

    def collate_fn(self, batch_list: List[TrainingInstance]) -> TrainingBatch:

        # Convert lists of ``image_id``s and ``caption_tokens``s as tensors.
        image_id = torch.tensor([instance["image_id"] for instance in batch_list]).long()
        caption_tokens = torch.tensor(
            [instance["caption_tokens"] for instance in batch_list]
        ).long()

        # Pad adaptive image features in the batch.
        image_features = torch.from_numpy(
            _collate_image_features([instance["image_features"] for instance in batch_list])
        )

        batch: TrainingBatch = {
            "image_id": image_id,
            "image_features": image_features,
            "caption_tokens": caption_tokens,
        }
        return batch


class ValidationDataset(Dataset):
    def __init__(self, image_features_h5path: str, in_memory: bool = True) -> None:

        self._image_features_reader = ImageFeaturesReader(image_features_h5path, in_memory)
        self._image_ids = sorted(list(self._image_features_reader._map.keys()))

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, index: int) -> ValidationInstance:
        image_id = self._image_ids[index]
        image_features = self._image_features_reader[image_id]

        item: ValidationInstance = {"image_id": image_id, "image_features": image_features}
        return item

    def collate_fn(self, batch_list: List[ValidationInstance]) -> ValidationBatch:
        # Convert lists of ``image_id``s and ``caption_tokens``s as tensors.
        image_id = torch.tensor([instance["image_id"] for instance in batch_list]).long()

        # Pad adaptive image features in the batch.
        image_features = torch.from_numpy(
            _collate_image_features([instance["image_features"] for instance in batch_list])
        )

        batch: ValidationBatch = {"image_id": image_id, "image_features": image_features}
        return batch


InferenceDataset = ValidationDataset
