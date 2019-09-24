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

from updown.types import ConstraintBoxes


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
        self._num_boxes: Dict[int, int] = {}

        if self._in_memory:
            print(f"Loading image features from {self.features_h5path}...")
            features_h5 = h5py.File(self.features_h5path, "r")

            # If loading all features in memory at once, keep a mapping of image id to features.
            for index in tqdm(range(features_h5["image_id"].shape[0])):
                self._map[features_h5["image_id"][index]] = features_h5["features"][index]
                self._num_boxes[features_h5["image_id"][index]] = features_h5["num_boxes"][index]

            features_h5.close()
        else:
            self.features_h5 = h5py.File(self.features_h5path, "r")
            image_id_np = np.array(self.features_h5["image_id"])

            # If not loading all features in memory at once, just keep a mapping of image id to
            # index of features in H5 file.
            self._map = {image_id_np[index]: index for index in range(image_id_np.shape[0])}

            # Load the number of boxes for each image anyway, there's not bulky.
            self._num_boxes = {
                image_id_np[index]: self.features_h5["num_boxes"][index]
                for index in range(image_id_np.shape[0])
            }

    def __len__(self) -> int:
        return len(self._map)

    def __getitem__(self, image_id: int) -> np.ndarray:
        if self._in_memory:
            image_id_features = self._map[image_id]
        else:
            index = self._map[image_id]
            image_id_features = self.features_h5["features"][index]

        num_boxes = self._num_boxes[image_id]
        return image_id_features.reshape((num_boxes, -1))


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

    def __len__(self) -> int:
        return len(self._captions)

    def __getitem__(self, index) -> Tuple[int, List[str]]:
        return self._captions[index]


class ConstraintBoxesReader(object):
    r"""
    A reader for annotation files containing detected bounding boxes (in COCO format). The JSON
    file should have ``categories``, ``images`` and ``annotations`` fields (similar to COCO
    instance annotations).

    Extended Summary
    ----------------
    For our use cases, the detections are from an object detector trained using Open Images.
    These can be produced for any set of images by following instructions
    `here <https://github.com/nocaps-org/image-feature-extractors#extract-boxes-from-oi-detector>`_.

    Parameters
    ----------
    boxes_jsonpath: str
        Path to a JSON file containing bounding box detections in COCO format (nocaps val/test
        usually).
    """

    def __init__(self, boxes_jsonpath: str):

        _boxes = json.load(open(boxes_jsonpath))

        # Form a mapping between Image ID and corresponding boxes from OI Detector.
        self._image_id_to_boxes: Dict[int, Any] = {}

        for ann in _boxes["annotations"]:
            if ann["image_id"] not in self._image_id_to_boxes:
                self._image_id_to_boxes[ann["image_id"]] = []

            self._image_id_to_boxes[ann["image_id"]].append(ann)

        # A list of Open Image object classes. Index of a class in this list is its Open Images
        # class ID. Open Images class IDs start from 1, so zero-th element is "__background__".
        self._class_names = [c["name"] for c in _boxes["categories"]]

    def __len__(self) -> int:
        return len(self._image_id_to_boxes)

    def __getitem__(self, image_id: int) -> ConstraintBoxes:

        # List of bounding box detections from OI detector in COCO format.
        # Some images may not have any boxes, handle that case too.
        bbox_anns = self._image_id_to_boxes.get(int(image_id), [])

        boxes = np.array([ann["bbox"] for ann in bbox_anns])
        scores = np.array([ann.get("score", 1) for ann in bbox_anns])

        # Convert object class IDs to their names.
        class_names = [self._class_names[ann["category_id"]] for ann in bbox_anns]

        return {"boxes": boxes, "class_names": class_names, "scores": scores}
