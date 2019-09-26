import csv
import json
from typing import Dict, List, Optional

import anytree
from anytree.search import findall
import numpy as np
import torch
from torchtext.vocab import GloVe
from allennlp.data import Vocabulary


def add_constraint_words_to_vocabulary(
    vocabulary: Vocabulary, wordforms_tsvpath: str, namespace: str = "tokens"
) -> Vocabulary:
    r"""
    Expand the :class:`~allennlp.data.vocabulary.Vocabulary` with CBS constraint words. We do not
    need to worry about duplicate words in constraints and caption vocabulary. AllenNLP avoids
    duplicates automatically.

    Parameters
    ----------
    vocabulary: allennlp.data.vocabulary.Vocabulary
        The vocabulary to be expanded with provided words.
    wordforms_tsvpath: str
        Path to a TSV file containing two fields: first is the name of Open Images object class
        and second field is a comma separated list of words (possibly singular and plural forms
        of the word etc.) which could be CBS constraints.
    namespace: str, optional (default="tokens")
        The namespace of :class:`~allennlp.data.vocabulary.Vocabulary` to add these words.

    Returns
    -------
    allennlp.data.vocabulary.Vocabulary
        The expanded :class:`~allennlp.data.vocabulary.Vocabulary` with all the words added.
    """

    with open(wordforms_tsvpath, "r") as wordforms_file:
        reader = csv.DictReader(wordforms_file, delimiter="\t", fieldnames=["class_name", "words"])
        for row in reader:
            for word in row["words"].split(","):
                # Constraint words can be "multi-word" (may have more than one tokens).
                # Add all tokens to the vocabulary separately.
                for w in word.split():
                    vocabulary.add_token_to_namespace(w, namespace)

    return vocabulary


class ConstraintFilter(object):

    # fmt: off
    BLACKLIST: List[str] = [
        "auto part", "bathroom accessory", "bicycle wheel", "boy", "building", "clothing",
        "door handle", "fashion accessory", "footwear", "girl", "hiking equipment", "human arm",
        "human beard", "human body", "human ear", "human eye", "human face", "human foot",
        "human hair", "human hand", "human head", "human leg", "human mouth", "human nose",
        "land vehicle", "mammal", "man", "person", "personal care", "plant", "plumbing fixture",
        "seat belt", "skull", "sports equipment", "tire", "tree", "vehicle registration plate",
        "wheel", "woman"
    ]
    # fmt: on

    REPLACEMENTS: Dict[str, str] = {
        "band-aid": "bandaid",
        "wood-burning stove": "wood burning stove",
        "kitchen & dining room table": "table",
        "salt and pepper shakers": "salt and pepper",
        "power plugs and sockets": "power plugs",
        "luggage and bags": "luggage",
    }

    def __init__(self, hierarchy_jsonpath: str, nms_threshold: float = 0.85, topk: int = 3):
        def __read_hierarchy(node: anytree.AnyNode, parent: Optional[anytree.AnyNode] = None):
            # Cast an ``anytree.AnyNode`` (after first level of recursion) to dict.
            attributes = dict(node)
            children = attributes.pop("Subcategory", [])

            node = anytree.AnyNode(parent=parent, **attributes)
            for child in children:
                __read_hierarchy(child, parent=node)
            return node

        # Read the object class hierarchy as a tree, to make searching easier.
        self._hierarchy = __read_hierarchy(json.load(open(hierarchy_jsonpath)))

        self._nms_threshold = nms_threshold
        self._topk = topk

    def __call__(self, boxes: np.ndarray, class_names: List[str], scores: np.ndarray) -> List[str]:

        # Remove padding boxes (which have prediction confidence score = 0), and remove boxes
        # corresponding to all blacklisted classes. These will never become CBS constraints.
        keep_indices = []
        for i in range(len(class_names)):
            if scores[i] > 0 and class_names[i] not in self.BLACKLIST:
                keep_indices.append(i)

        boxes = boxes[keep_indices]
        class_names = [class_names[i] for i in keep_indices]
        scores = scores[keep_indices]

        # Perform non-maximum suppression according to category hierarchy. For example, for highly
        # overlapping boxes on a dog, "dog" suppresses "animal".
        keep_indices = self._nms(boxes, class_names)
        boxes = boxes[keep_indices]
        class_names = [class_names[i] for i in keep_indices]
        scores = scores[keep_indices]

        # Retain top-k constraints based on prediction confidence score.
        class_names_and_scores = sorted(list(zip(class_names, scores)), key=lambda t: -t[1])
        class_names_and_scores = class_names_and_scores[: self._topk]

        # Replace class name according to ``self.REPLACEMENTS``.
        class_names = [self.REPLACEMENTS.get(t[0], t[0]) for t in class_names_and_scores]

        # Drop duplicates.
        class_names = list(set(class_names))
        return class_names

    def _nms(self, boxes: np.ndarray, class_names: List[str]):
        r"""
        Perform non-maximum suppression of overlapping boxes, where the score is based on "height"
        of class in the hierarchy.
        """

        if len(class_names) == 0:
            return []

        # For object class, get the height of its corresponding node in the hierarchy tree.
        # Less height => finer-grained class name => higher score.
        heights = np.array(
            [
                findall(self._hierarchy, lambda node: node.LabelName.lower() in c)[0].height
                for c in class_names
            ]
        )
        # Get a sorting of the heights in ascending order, i.e. higher scores first.
        score_order = heights.argsort()

        # Compute areas for calculating intersection over union. Add 1 to avoid division by zero
        # for zero area (padding/dummy) boxes.
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Fill "keep_boxes" with indices of boxes to keep, move from left to right in
        # ``score_order``, keep current box index (score_order[0]) and suppress (discard) other
        # indices of boxes having lower IoU threshold with current box from ``score_order``.
        # list. Note the order is a sorting of indices according to scores.
        keep_box_indices = []

        while score_order.size > 0:
            # Keep the index of box under consideration.
            current_index = score_order[0]
            keep_box_indices.append(current_index)

            # For the box we just decided to keep (score_order[0]), compute its IoU with other
            # boxes (score_order[1:]).
            xx1 = np.maximum(x1[score_order[0]], x1[score_order[1:]])
            yy1 = np.maximum(y1[score_order[0]], y1[score_order[1:]])
            xx2 = np.minimum(x2[score_order[0]], x2[score_order[1:]])
            yy2 = np.minimum(y2[score_order[0]], y2[score_order[1:]])

            intersection = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
            union = areas[score_order[0]] + areas[score_order[1:]] - intersection

            # Perform NMS for IoU >= 0.85. Check score, boxes corresponding to object
            # classes with smaller/equal height in hierarchy cannot be suppressed.
            keep_condition = np.logical_or(
                heights[score_order[1:]] >= heights[score_order[0]],
                intersection / union <= self._nms_threshold,
            )

            # Only keep the boxes under consideration for next iteration.
            score_order = score_order[1:]
            score_order = score_order[np.where(keep_condition)[0]]

        return keep_box_indices


class FiniteStateMachineBuilder(object):
    def __init__(self, vocabulary: Vocabulary, wordforms_tsvpath: str, max_num_states: int = 26):
        self._vocabulary = vocabulary
        self._max_num_states = max_num_states

        self._wordforms: Dict[str, List[str]] = {}
        with open(wordforms_tsvpath, "r") as wordforms_file:
            reader = csv.DictReader(
                wordforms_file, delimiter="\t", fieldnames=["class_name", "words"]
            )
            for row in reader:
                self._wordforms[row["class_name"]] = row["words"].split(",")

    @staticmethod
    def _connect(
        fsm: torch.Tensor,
        from_state: int,
        to_state: int,
        word_indices: List[int],
        reset_state: int = None,
    ):
        for word_index in word_indices:
            fsm[from_state, to_state, word_index] = 1
            fsm[from_state, from_state, word_index] = 0

            if reset_state is not None:
                fsm[from_state, from_state, :] = 0
                fsm[from_state, reset_state, :] = 1
                fsm[from_state, reset_state, word_index] = 0

        return fsm

    def build(self, candidates: List[str]):
        fsm = (
            torch.eye(self._max_num_states, dtype=torch.uint8)
            .unsqueeze(-1)
            .repeat(1, 1, self._vocabulary.get_vocab_size())
        )
        last_state = 8
        level_mapping = [{3: 5, 2: 6}, {1: 6, 3: 4}, {1: 5, 2: 4}]
        for i, candidate in enumerate(candidates):
            class_words = candidate.split()

            if len(class_words) == 1:
                group_s1 = [
                    self._vocabulary.get_token_index(wf) for wf in self._wordforms[class_words[0]]
                ]

                fsm = self._connect(fsm, 0, i + 1, group_s1)
                fsm = self._connect(fsm, i + 4, 7, group_s1)

                mapping = level_mapping[i]
                for j in range(1, 4):
                    if j in mapping:
                        fsm = self._connect(fsm, j, mapping[j], group_s1)
            elif len(class_words) == 2:
                [word1, word2] = class_words
                group_s1 = [self._vocabulary.get_token_index(wf) for wf in self._wordforms[word1]]
                group_s2 = [self._vocabulary.get_token_index(wf) for wf in self._wordforms[word2]]

                fsm = self._connect(fsm, 0, last_state, group_s1)
                fsm = self._connect(fsm, last_state, i + 1, group_s2, reset_state=0)
                last_state += 1

                fsm = self._connect(fsm, i + 4, last_state, group_s1)
                fsm = self._connect(fsm, last_state, 7, group_s2, reset_state=i + 4)
                last_state += 1

                mapping = level_mapping[i]
                for j in range(1, 4):
                    if j in mapping:
                        fsm = self._connect(fsm, j, last_state, group_s1)
                        fsm = self._connect(fsm, last_state, mapping[j], group_s2, reset_state=j)
                        last_state += 1
            elif len(class_words) == 3:
                [word1, word2, word3] = class_words
                group_s1 = [self._vocabulary.get_token_index(wf) for wf in self._wordforms[word1]]
                group_s2 = [self._vocabulary.get_token_index(wf) for wf in self._wordforms[word2]]
                group_s3 = [self._vocabulary.get_token_index(wf) for wf in self._wordforms[word3]]

                fsm = self._connect(fsm, 0, last_state, group_s1)
                fsm = self._connect(fsm, last_state, last_state + 1, group_s2, reset_state=0)
                fsm = self._connect(fsm, last_state + 1, i + 1, group_s3, reset_state=0)
                last_state += 2

                fsm = self._connect(fsm, i + 4, last_state, group_s1)
                fsm = self._connect(fsm, last_state, last_state + 1, group_s2, reset_state=i + 4)
                fsm = self._connect(fsm, last_state + 1, 7, group_s3, reset_state=i + 4)
                last_state += 2

                mapping = level_mapping[i]
                for j in range(1, 4):
                    if j in mapping:
                        fsm = self._connect(fsm, j, last_state, group_s1)
                        fsm = self._connect(
                            fsm, last_state, last_state + 1, group_s2, reset_state=j
                        )
                        fsm = self._connect(
                            fsm, last_state + 1, mapping[j], group_s3, reset_state=j
                        )
                        last_state += 2

        return fsm, last_state


def cbs_select_best_beam(
    beams: torch.Tensor,
    beam_log_probabilities: torch.Tensor,
    given_constraints: torch.Tensor,
    min_constraints_to_satisfy: int = 2,
):
    r"""
    Select the best beam decoded with the highest likelihood, and which also satisfies specified
    minimum constraints out of a total number of given constraints.

    .. note::

        The implementation of this function goes hand-in-hand with the FSM building implementation
        in :meth:`~updown.utils.cbs.FiniteStateMachineBuilder.build`, which specifies which state
        satisfies which (basically, how many) constraints. If the "definition" of states change,
        then selection of beams also changes accordingly.

    Parameters
    ----------
    beams: torch.Tensor
        A tensor of shape ``(batch_size, num_states, beam_size, max_decoding_steps)`` containing
        decoded beams by :class:`~updown.modules.cbs.ConstrainedBeamSearch`. These beams are
        sorted according to their likelihood (descending) in ``beam_size`` dimension.
        already best beams in their state.
    beam_log_probabilities: torch.Tensor
        A tensor of shape ``(batch_size, num_states, beam_size)`` containing likelihood of decoded
        beams.
    given_constraints: torch.Tensor
        A tensor of shape ``(batch_size, )`` containing number of constraints given at the start
        of decoding.
    min_constraints_to_satisfy: int, optional (default = 2)
        Minimum number of constraints to satisfy. This is either 2, or ``given_constraints`` if
        they are less than 2. Beams corresponding to states not satisfying at least these number
        of constraints will be dropped. Only up to 3 supported.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple with two elements: decoded sequence (be am) which has highest likelihood among
        beams satisfying constraints, and the likelihood value of chosen beam.
    """
    if min_constraints_to_satisfy > 3:
        raise ValueError(
            f"Cannot satisfy {min_constraints_to_satisfy} constraints. Only up to 3 supported."
        )

    batch_size, num_states, beam_size, max_decoding_steps = beams.size()

    # Detach so the graph goes out of scope and avoid memory leak.
    beams = beams.detach()

    best_beams: List[torch.Tensor] = []
    best_beam_log_probabilities: List[torch.Tensor] = []

    for i in range(batch_size):
        given_constraints_for_this = given_constraints[i].item()

        min_constraints_to_satisfy_for_this = min(
            min_constraints_to_satisfy, given_constraints_for_this
        )

        # For given constraints (a,b,c) and states S[0:8] -
        if min_constraints_to_satisfy_for_this == 0:
            # Just select state 0 (no constraints)
            best_beams.append(beams[i, 0, 0, :])
            best_beam_log_probabilities.append(beam_log_probabilities[i, 0, 0])

        elif min_constraints_to_satisfy_for_this == 1:
            # One constraint given, select state 1 (a).
            best_beams.append(beams[i, 1, 0, :])
            best_beam_log_probabilities.append(beam_log_probabilities[i, 1, 0])

        elif min_constraints_to_satisfy_for_this >= 2:
            # According to the FSM builder logic, states satisfying two or more constraints are
            # S4 (b,c), S5 (a,c), S6 (a,b), S7 (a,b,c). "c" might be a dummy constraint.

            selected_indices = torch.argmax(beam_log_probabilities[:, 4:8, 0], dim=1)
            selected_indices += (
                torch.arange(
                    selected_indices.size(0),
                    device=selected_indices.device,
                    dtype=selected_indices.dtype,
                )
                * 4
            )
            top_beams = beams[:, 4:8, 0, :].contiguous().view(-1, max_decoding_steps)
            top_beams = top_beams.index_select(0, selected_indices)

            top_beam_logprobs = beam_log_probabilities[:, 4:8, 0].contiguous().view(-1)
            top_beam_logprobs = top_beam_logprobs.index_select(0, selected_indices)

            best_beams.append(top_beams[i])
            best_beam_log_probabilities.append(top_beam_logprobs[i])

    # shape: (batch_size, max_decoding_steps), (batch_size, )
    return (
        torch.stack(best_beams).long().to(beam_log_probabilities.device),
        torch.stack(best_beam_log_probabilities).to(beam_log_probabilities.device),
    )
