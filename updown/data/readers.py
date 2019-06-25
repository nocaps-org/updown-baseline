r"""
A Reader simply reads data from disk and returns it _almost_ as is. Readers should be
utilized by PyTorch :class:`~torch.utils.data.Dataset`. Too much of data pre-processing is not
recommended in the reader, such as tokenizing words to integers, embedding tokens, or passing
an image through a pre-trained CNN. Each reader must implement at least two methods:

    1. ``__len__`` to return the length of data this Reader can read.
    2. ``__getitem__`` to return data based on an index or a primary key (such as ``image_id``).
"""
import json
from typing import Any, Dict, List, Tuple, Union

import h5py
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm


class ImageFeaturesReader(object):
    r"""
    A reader for H5 files containing pre-extracted image features. A typical image features file
    should have at least two H5 datasets, named ``image_id`` and ``features``. It may optionally
    have other H5 datasets, such as ``boxes`` (for bounding box coordinates), ``width`` and
    ``height`` for image size, and others. This reader only reads image features, because our
    UpDown captioner baseline does not require anything other than image features.

    Example of an h5 file::

        image_bottomup_features.h5
        |--- "image_id" [shape: (num_images, )]
        |--- "features" [shape: (num_images, num_boxes, feature_size)]
        +--- .attrs {"split": "coco_train2017"}

    Parameters
    ----------
    features_h5path : str
        Path to an H5 file containing image ids and features corresponding to one of the four
        splits used: "coco_train2017", "coco_val2017", "nocaps_val", "nocaps_test".
    in_memory : bool
        Whether to load the features in memory. Beware, these files are sometimes tens of GBs
        in size. Set this to true if you have sufficient RAM.
    """

    def __init__(self, features_h5path: str, in_memory: bool = False) -> None:
        self.features_h5path = features_h5path
        self._in_memory = in_memory

        # Keys are all the image ids, values depend on ``self._in_memory``.
        # If ``self._in_memory`` is True, values are image features corresponding to the image id.
        # Else values will be integers; indices in the files to read features from.
        self._map: Dict[int, Union[int, np.ndarray]] = {}

        if self._in_memory:
            print(f"Loading image features from {self.features_h5path}...")
            features_h5 = h5py.File(self.features_h5path, "r")

            # If loading all features in memory at once, keep a mapping of image id to features.
            for index in tqdm(range(features_h5["image_id"].shape[0])):
                self._map[features_h5["image_id"][index]] = features_h5["features"][index]
            features_h5.close()

        else:
            self.features_h5 = h5py.File(self.features_h5path, "r")
            image_id_np = np.array(self.features_h5["image_id"])

            # If not loading all features in memory at once, just keep a mapping of image id to
            # index of features in H5 file.
            self._map = {
                image_id_np[index]: index for index in range(image_id_np.shape[0])
            }

    def __len__(self) -> int:
        return len(self._map)

    def __getitem__(self, image_id: int) -> np.ndarray:
        if self._in_memory:
            return self._map[image_id]
        else:
            index = self._map[image_id]
            image_id_features = self.features_h5["features"][index]
            return image_id_features


class CocoCaptionsReader(object):
    r"""
    A reader for annotation files containing training captions. These are JSON files in COCO
    format.

    Parameters
    ----------
    captions_jsonpath : str
        Path to a JSON file containing training captions in COCO format (COCO train2017 usually).
    """
    def __init__(self, captions_jsonpath: str) -> None:
        self._captions_jsonpath = captions_jsonpath

        with open(self._captions_jsonpath) as cap:
            captions_json: Dict[str, Any] = json.load(cap)
        # fmt: off
        # List of punctuations taken from pycocoevalcap - these are ignored during evaluation.
        PUNCTUATIONS: List[str] = [
            "''", "'", "``", "`", "(", ")", "{", "}",
            ".", "?", "!", ",", ":", "-", "--", "...", ";"
        ]
        # fmt: on

        # List of (image id, caption) tuples.
        self._captions: List[Tuple[int, List[str]]] = []

        print(f"Tokenizing captions from {captions_jsonpath}...")
        for caption_item in tqdm(captions_json["annotations"]):

            caption: str = caption_item["caption"].lower().strip()
            caption_tokens: List[str] = word_tokenize(caption)
            caption_tokens = [ct for ct in caption_tokens if ct not in PUNCTUATIONS]

            self._captions.append((caption_item["image_id"], caption_tokens))

    def __len__(self):
        return len(self._captions)

    def __getitem__(self, index) -> Tuple[int, List[str]]:
        return self._captions[index]
