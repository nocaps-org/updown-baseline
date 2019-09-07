import numpy as np
import torch
from allennlp.data import Vocabulary
import h5py
import json

from anytree import AnyNode
from anytree.search import findall

BLACKLIST_CATEGORIES = [
    "Tree",
    "Building",
    "Plant",
    "Man",
    "Woman",
    "Person",
    "Boy",
    "Girl",
    "Human eye",
    "Skull",
    "Human head",
    "Human face",
    "Human mouth",
    "Human ear",
    "Human nose",
    "Human hair",
    "Human hand",
    "Human foot",
    "Human arm",
    "Human leg",
    "Human beard",
    "Human body",
    "Vehicle registration plate",
    "Wheel",
    "Seat belt",
    "Tire",
    "Bicycle wheel",
    "Auto part",
    "Door handle",
    "Clothing",
    "Footwear",
    "Fashion accessory",
    "Sports equipment",
    "Hiking equipment",
    "Mammal",
    "Personal care",
    "Bathroom accessory",
    "Plumbing fixture",
    "Land vehicle",
]

replacements = {
    "band-aid": "bandaid",
    "wood-burning stove": "wood burning stove",
    "kitchen & dining room table": "table",
    "salt and pepper shakers": "salt and pepper",
    "power plugs and sockets": "power plugs",
    "luggage and bags": "luggage",
}


class OIDictImporter(object):
    """ Importer that works on Open Images json hierarchy """

    def __init__(self, nodecls=AnyNode):
        self.nodecls = nodecls

    def import_(self, data):
        """Import tree from `data`."""
        return self.__import(data)

    def __import(self, data, parent=None):
        assert isinstance(data, dict)
        assert "parent" not in data
        attrs = dict(data)
        children = attrs.pop("Subcategory", []) + attrs.pop("Part", [])
        node = self.nodecls(parent=parent, **attrs)
        for child in children:
            self.__import(child, parent=node)
        return node


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


def nms(dets, classes, hierarchy, thresh=0.7):
    # Non-max suppression of overlapping boxes where score is based on 'height' in the hierarchy,
    # defined as the number of edges on the longest path to a leaf

    scores = [
        findall(hierarchy, filter_=lambda node: node.LabelName in (cls))[0].height
        for cls in classes
    ]

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    scores = np.array(scores)
    order = scores.argsort()

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # check the score, objects with smaller or equal number of layers cannot be removed.
        keep_condition = np.logical_or(
            scores[order[1:]] <= scores[i], inter / (areas[i] + areas[order[1:]] - inter) <= thresh
        )

        inds = np.where(keep_condition)[0]
        order = order[inds + 1]

    return keep


class CBSConstraint(object):
    def __init__(
        self,
        features_h5path: str,
        oi_class_path: str,
        oi_word_form_path: str,
        class_structure_json_path: str,
        vocabulary: Vocabulary,
        topk: int = 3,
    ):
        self.features_h5path = features_h5path
        self.topk = topk
        self._vocabulary = vocabulary
        self._pad_index = vocabulary.get_token_index("@@UNKNOWN@@")
        self.M = _CBSMatrix(self._vocabulary.get_vocab_size())

        self.boxes_h5 = h5py.File(self.features_h5path, "r")
        image_id_np = np.array(self.boxes_h5["image_id"])
        self._map = {image_id_np[index]: index for index in range(image_id_np.shape[0])}

        self.oi_class_list = [None]
        with open(oi_class_path) as out:
            for line in out:
                self.oi_class_list.append(line.strip().split(",")[1])

        oov, total = 0, 0
        self.oi_word_form = {}
        with open(oi_word_form_path) as out:
            for line in out:
                line = line.strip()
                items = line.split("\t")
                w_list = items[1].split(",")
                self.oi_word_form[items[0]] = w_list

                for w in w_list:
                    for ch in w.split():
                        ch_index = self._vocabulary.get_token_index(ch)
                        oov += 1 if ch_index == self._pad_index else 0
                    total += len(w.split())
        print("object class word OOV %d / %d = %.2f" % (oov, total, 100 * oov / total))

        self.obj_num = {}

        importer = OIDictImporter()
        with open(class_structure_json_path) as f:
            self.class_structure = importer.import_(json.load(f))

    def select_state_func(self, beam_prediction, beam_score, imageID):
        max_step = beam_prediction.size(-1)
        selected_indices = torch.argmax(beam_score[:, 4:8, 0], dim=1)
        selected_indices += (
            torch.arange(
                selected_indices.size(0),
                device=selected_indices.device,
                dtype=selected_indices.dtype,
            )
            * 4
        )
        top_two_beam_prediction = beam_prediction[:, 4:8, 0, :].contiguous().view(-1, max_step)
        top_two_beam_prediction = top_two_beam_prediction.index_select(0, selected_indices)

        top_two_beam_prediction = top_two_beam_prediction.cpu().detach().numpy()
        beam_prediction = beam_prediction.cpu().detach().numpy()
        imageID = imageID.cpu().detach().numpy().tolist()

        pred = []
        for i, ID in enumerate(imageID):
            label_num = self.obj_num[ID]
            if label_num >= 2:
                # Two labels must be satisfied together
                pred.append(top_two_beam_prediction[i])
            elif label_num >= 1:
                # one label must be satisfied together
                pred.append(beam_prediction[i, 1, 0, :])
            else:
                # no constraint, get original beam
                pred.append(beam_prediction[i, 0, 0, :])

        return torch.from_numpy(np.array(pred, dtype=np.int64)).to(beam_score.device)

    def get_word_set(self, target):
        if target in self.oi_word_form:
            group_w = self.oi_word_form[target]
        else:
            group_w = [target]

        group_w = [self._vocabulary.get_token_index(w) for w in group_w]
        return [v for v in group_w if not (v == self._pad_index)]

    def get_state_matrix(self, image_id: int):
        i = self._map[image_id.item()]

        box = self.boxes_h5["boxes"][i]
        box_cls = self.boxes_h5["classes"][i]
        box_score = self.boxes_h5["scores"][i]

        keep = suppress_parts(box_score, [self.oi_class_list[cls_] for cls_ in box_cls])
        box = box[keep]
        box_cls = box_cls[keep]
        box_score = box_score[keep]

        keep = nms(box, [self.oi_class_list[cls_] for cls_ in box_cls], self.class_structure)
        box = box[keep]
        box_cls = box_cls[keep]
        box_score = box_score[keep]

        anns = list(zip(box_score, box_cls))
        anns = sorted(anns, key=lambda x: x[0], reverse=True)

        candidates = []
        for s, cls_idx in anns[: self.topk]:  # Keep up to three classes
            text = self.oi_class_list[cls_idx].lower()
            if text in replacements:
                text = replacements[text]
            if text not in candidates:
                candidates.append(text)

        self.obj_num[image_id.item()] = len(candidates)

        self.M.init_matrix(26)
        for i in range(26):
            self.M.init_row(i)

        start_addtional_index = 8
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

                self.M.add_connect(0, start_addtional_index, group_s1)
                self.M.add_connect(start_addtional_index, i + 1, group_s2)
                start_addtional_index += 1

                self.M.add_connect(i + 4, start_addtional_index, group_s1)
                self.M.add_connect(start_addtional_index, 7, group_s2)
                start_addtional_index += 1

                mapping = level_mapping[i]
                for j in range(1, 4):
                    if j in mapping:
                        self.M.add_connect(j, start_addtional_index, group_s1)
                        self.M.add_connect(start_addtional_index, mapping[j], group_s2)
                        start_addtional_index += 1
            elif len(word_list) == 3:
                [s1, s2, s3] = word_list
                group_s1 = self.get_word_set(s1)
                group_s2 = self.get_word_set(s2)
                group_s3 = self.get_word_set(s3)

                self.M.add_connect(0, start_addtional_index, group_s1)
                self.M.add_connect(start_addtional_index, start_addtional_index + 1, group_s2)
                self.M.add_connect(start_addtional_index + 1, i + 1, group_s3)
                start_addtional_index += 2

                self.M.add_connect(i + 4, start_addtional_index, group_s1)
                self.M.add_connect(start_addtional_index, start_addtional_index + 1, group_s2)
                self.M.add_connect(start_addtional_index + 1, 7, group_s3)
                start_addtional_index += 2

                mapping = level_mapping[i]
                for j in range(1, 4):
                    if j in mapping:
                        self.M.add_connect(j, start_addtional_index, group_s1)
                        self.M.add_connect(
                            start_addtional_index, start_addtional_index + 1, group_s2
                        )
                        self.M.add_connect(start_addtional_index + 1, mapping[j], group_s3)
                        start_addtional_index += 2

        return self.M.matrix, start_addtional_index


class FreeConstraint(object):
    def __init__(self, output_size):
        self.M = _CBSMatrix(output_size)

    def select_state_func(self, beam_prediction, beam_score, imageID):
        return beam_prediction[:, 0, 0]

    def get_state_matrix(self, image_id):
        self.M.init_matrix(1)
        self.M.init_row(0)
        return self.M.matrix, 1
