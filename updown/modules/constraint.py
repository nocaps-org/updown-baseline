import json
from typing import Any, Dict, List

import h5py
import numpy as np
from allennlp.data import Vocabulary

from updown.utils.cbs import ConstraintFilter


class _CBSMatrix(object):
    def __init__(self, vocab_size: int):
        self._matrix = None
        self.vocab_size = vocab_size

    def init_matrix(self, state_size):
        self._matrix = np.zeros((1, state_size, state_size, self.vocab_size), dtype=np.uint8)

    def add_connect(self, from_state, to_state, w_group):
        assert self._matrix is not None
        for w_index in w_group:
            self._matrix[0, from_state, to_state, w_index] = 1
            self._matrix[0, from_state, from_state, w_index] = 0

    def init_row(self, state_index):
        assert self._matrix is not None
        self._matrix[0, state_index, state_index, :] = 1

    @property
    def matrix(self):
        return self._matrix


def suppress_parts(scores, classes):
    # just remove those 39 words
    keep = [
        i
        for i, (cls, score) in enumerate(zip(classes, scores))
        if score > 0.01 and cls not in BLACKLIST_CATEGORIES
    ]
    return keep


class CBSConstraint(object):
    def __init__(
        self,
        vocabulary: Vocabulary,
        boxes_jsonpath: str,
        oi_word_form_path: str,
        hierarchy_jsonpath: str,
        nms_threshold: float = 0.85,
        topk: int = 3,
    ):
        self._vocabulary = vocabulary
        self._pad_index = vocabulary.get_token_index("@@UNKNOWN@@")
        self.M = _CBSMatrix(self._vocabulary.get_vocab_size())

        _boxes = json.load(open(boxes_jsonpath))

        # Form a mapping between Image ID and corresponding boxes from OI Detector.
        self._image_id_to_boxes: Dict[int, Any] = {}

        for ann in _boxes["annotations"]:
            if ann["image_id"] not in self._image_id_to_boxes:
                self._image_id_to_boxes[ann["image_id"]] = []

            self._image_id_to_boxes[ann["image_id"]].append(ann)

        # A list of Open Image object classes. Index of a class in this list is its Open Images
        # class ID. Open Images class IDs start from 1, so zero-th element is "__background__".
        self.oi_class_list = [c["name"] for c in _boxes["categories"]]

        self.oi_word_form: Dict[str, List[str]] = {}
        with open(oi_word_form_path) as out:
            for line in out:
                line = line.strip()
                items = line.split("\t")
                w_list = items[1].split(",")
                self.oi_word_form[items[0]] = w_list

        self._constraint_filter = ConstraintFilter(hierarchy_jsonpath, nms_threshold, topk)

    def get_word_set(self, target):
        if target in self.oi_word_form:
            group_w = self.oi_word_form[target]
        else:
            group_w = [target]

        group_w = [self._vocabulary.get_token_index(w) for w in group_w]
        return [v for v in group_w if not (v == self._pad_index)]

    def get_state_matrix(self, image_id):

        # List of bounding box detections from OI detector in COCO format.
        bbox_anns = self._image_id_to_boxes[int(image_id)]

        boxes = np.array([ann["bbox"] for ann in bbox_anns])
        class_ids = np.array([ann["category_id"] for ann in bbox_anns])
        scores = np.array([ann.get("score", 1) for ann in bbox_anns])

        # Convert object class IDs to their names.
        class_names = [self.oi_class_list[c] for c in class_ids]

        # Apply constraint filtering to object classes.
        candidates = self._constraint_filter(boxes, class_names, scores)

        self.M.init_matrix(26)
        for i in range(26):
            self.M.init_row(i)

        start_additional_index = 8
        level_mapping = [{3: 5, 2: 6}, {1: 6, 3: 4}, {1: 5, 2: 4}]
        for i, target in enumerate(candidates):
            word_list = target.split()

            if len(word_list) == 1:
                group_w = self.get_word_set(target)

                self.M.add_connect(0, i + 1, group_w)
                self.M.add_connect(i + 4, 7, group_w)

                mapping = level_mapping[i]
                for j in range(1, 4):
                    if j in mapping:
                        self.M.add_connect(j, mapping[j], group_w)
            elif len(word_list) == 2:
                [s1, s2] = word_list
                group_s1 = self.get_word_set(s1)
                group_s2 = self.get_word_set(s2)

                self.M.add_connect(0, start_additional_index, group_s1)
                self.M.add_connect(start_additional_index, i + 1, group_s2)
                start_additional_index += 1

                self.M.add_connect(i + 4, start_additional_index, group_s1)
                self.M.add_connect(start_additional_index, 7, group_s2)
                start_additional_index += 1

                mapping = level_mapping[i]
                for j in range(1, 4):
                    if j in mapping:
                        self.M.add_connect(j, start_additional_index, group_s1)
                        self.M.add_connect(start_additional_index, mapping[j], group_s2)
                        start_additional_index += 1
            elif len(word_list) == 3:
                [s1, s2, s3] = word_list
                group_s1 = self.get_word_set(s1)
                group_s2 = self.get_word_set(s2)
                group_s3 = self.get_word_set(s3)

                self.M.add_connect(0, start_additional_index, group_s1)
                self.M.add_connect(start_additional_index, start_additional_index + 1, group_s2)
                self.M.add_connect(start_additional_index + 1, i + 1, group_s3)
                start_additional_index += 2

                self.M.add_connect(i + 4, start_additional_index, group_s1)
                self.M.add_connect(start_additional_index, start_additional_index + 1, group_s2)
                self.M.add_connect(start_additional_index + 1, 7, group_s3)
                start_additional_index += 2

                mapping = level_mapping[i]
                for j in range(1, 4):
                    if j in mapping:
                        self.M.add_connect(j, start_additional_index, group_s1)
                        self.M.add_connect(
                            start_additional_index, start_additional_index + 1, group_s2
                        )
                        self.M.add_connect(start_additional_index + 1, mapping[j], group_s3)
                        start_additional_index += 2

        return self.M.matrix, start_additional_index, len(candidates)


class FreeConstraint(object):
    def __init__(self, output_size):
        self.M = _CBSMatrix(output_size)

    def get_state_matrix(self, image_id):
        self.M.init_matrix(1)
        self.M.init_row(0)
        return self.M.matrix, 1, 0
