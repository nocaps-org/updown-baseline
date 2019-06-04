from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from allennlp.data import Vocabulary

from updown.readers import CocoCaptionsReader, ImageFeaturesReader


class CocoTrainDataset(Dataset):
    def __init__(
        self,
        vocabulary: Vocabulary,
        captions_jsonpath: str,
        image_features_h5path: str,
        max_caption_length: int = 20,
        in_memory: bool = True,
    ):

        self._vocabulary = vocabulary
        self._captions_reader = CocoCaptionsReader(captions_jsonpath)
        self._image_features_reader = ImageFeaturesReader(image_features_h5path, in_memory)

        self._max_caption_length = max_caption_length

    def __len__(self):
        # Number of training examples are number of captions, not number of images.
        return len(self._captions_reader)

    def __getitem__(self, index: int) -> Dict[str, Any]:

        image_id, caption = self._captions_reader[index]
        image_features = self._image_features_reader[image_id]

        # Tokenize caption.
        caption_tokens: List[int] = [self._vocabulary.get_token_index(c) for c in caption]

        # Pad upto max_caption_length.
        caption_tokens = caption_tokens[: self._max_caption_length]
        caption_tokens.extend(
            [self._vocabulary.get_token_index("@@PADDING@@")]
            * (self._max_caption_length - len(caption_tokens))
        )

        item: Dict[str, Any] = {
            "image_id": torch.tensor(image_id).long(),
            "image_features": torch.tensor(image_features),
            "caption_tokens": torch.tensor(caption_tokens).long(),
        }
        return item


class InferenceDataset(Dataset):
    def __init__(self, image_features_h5path: str, in_memory: bool = True):

        self._image_features_reader = ImageFeaturesReader(image_features_h5path, in_memory)
        self._image_ids = sorted(list(self._image_features_reader._map.keys()))

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:

        image_id, = self._image_ids[index]
        image_features = self._image_features_reader[image_id]

        item: Dict[str, Any] = {
            "image_id": torch.tensor(image_id).long(),
            "image_features": torch.tensor(image_features),
        }
        return item
