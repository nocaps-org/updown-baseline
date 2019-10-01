r"""
A collection of helper methods and classes for preparing constraints and finite state machine
for performing Constrained Beam Search.
"""
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
    r"""
    A helper class to perform constraint filtering for providing sensible set of constraint words
    while decoding.

    Extended Summary
    ----------------
    The original work proposing `Constrained Beam Search <https://arxiv.org/abs/1612.00576>`_
    selects constraints randomly.

    We remove certain categories from a fixed set of "blacklisted" categories, which are either
    too rare, not commonly uttered by humans, or well covered in COCO. We resolve overlapping
    detections (IoU >= 0.85) by removing the higher-order of the two objects (e.g. , a "dog" would
    suppress a ‘mammal’) based on the Open Images class hierarchy (keeping both if equal).
    Finally, we take the top-k objects based on detection confidence as constraints.

    Parameters
    ----------
    hierarchy_jsonpath: str
        Path to a JSON file containing a hierarchy of Open Images object classes.
    nms_threshold: float, optional (default = 0.85)
        NMS threshold for suppressing generic object class names during constraint filtering,
        for two boxes with IoU higher than this threshold, "dog" suppresses "animal".
    max_given_constraints: int, optional (default = 3)
        Maximum number of constraints which can be specified for CBS decoding. Constraints are
        selected based on the prediction confidence score of their corresponding bounding boxes.
    """

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

    def __init__(
        self, hierarchy_jsonpath: str, nms_threshold: float = 0.85, max_given_constraints: int = 3
    ):
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
        self._max_given_constraints = max_given_constraints

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
        class_names_and_scores = class_names_and_scores[: self._max_given_constraints]

        # Replace class name according to ``self.REPLACEMENTS``.
        class_names = [self.REPLACEMENTS.get(t[0], t[0]) for t in class_names_and_scores]

        # Drop duplicates.
        class_names = list(set(class_names))
        return class_names

    def _nms(self, boxes: np.ndarray, class_names: List[str]):
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
    r"""
    A helper class to build a Finite State Machine for Constrained Beam Search, as per the
    state transitions shown in Figures 7 through 9 from our
    `paper appendix <https://arxiv.org/abs/1812.08658>`_.

    The FSM is constructed on a per-example basis, and supports up to three constraints,
    with each constraint being an Open Image class having up to three words (for example
    ``salt and pepper``). Each word in the constraint may have several word-forms (for
    example ``dog``, ``dogs``).

    .. note:: Providing more than three constraints may work but it is not tested.

    **Details on Finite State Machine Representation**

    .. image:: ../_static/fsm.jpg

    The FSM is representated as an adjacency matrix. Specifically, it is a tensor of shape
    ``(num_total_states, num_total_states, vocab_size)``. In this, ``fsm[S1, S2, W] = 1`` indicates
    a transition from "S1" to "S2" if word "W" is decoded. For example, consider **Figure 9**.
    The decoding is at initial state (``q0``), constraint word is ``D1``, while any other word
    in the vocabulary is ``Dx``. Then we have::

        fsm[0, 0, D1] = 0 and fsm[0, 1, D1] = 1    # arrow from q0 to q1
        fsm[0, 0, Dx] = 1 and fsm[0, 1, Dx] = 0    # self-loop on q0

    Consider up to "k" (3) constraints and up to "w" (3) words per constraint. We define these
    terms (as members in the class).

    .. code-block::

        _num_main_states = 2 ** k (8)
        _total_states = num_main_states * w (24)

    First eight states are considered as "main states", and will always be a part of the FSM. For
    less than "k" constraints, some states will be unreachable, hence "useless". These will be
    ignored automatically.

    For any multi-word constraint, we use extra "sub-states" after first ``2 ** k`` states. We
    make connections according to **Figure 7-8** for such constraints. We dynamically trim unused
    sub-states to save computation during decoding. That said, ``num_total_states`` dimension is
    at least 8.

    A state "q" satisfies number of constraints equal to the number of "1"s in the binary
    representation of that state. For example:

      - state "q0" (000) satisfies 0 constraints.
      - state "q1" (001) satisfies 1 constraint.
      - state "q2" (010) satisfies 1 constraint.
      - state "q3" (011) satisfies 2 constraints.

    and so on. Only main states fully satisfy constraints.

    Parameters
    ----------
    vocabulary: allennlp.data.Vocabulary
        AllenNLP’s vocabulary containing token to index mapping for captions vocabulary.
    wordforms_tsvpath: str
        Path to a TSV file containing two fields: first is the name of Open Images object class
        and second field is a comma separated list of words (possibly singular and plural forms
        of the word etc.) which could be CBS constraints.
    max_given_constraints: int, optional (default = 3)
        Maximum number of constraints which could be given while cbs decoding. Up to three
        supported.
    max_words_per_constraint: int, optional (default = 3)
        Maximum number of words per constraint for multi-word constraints. Note that these are
        for multi-word object classes (for example: ``fire hydrant``) and not for multiple
        "word-forms" of a word, like singular-plurals. Up to three supported.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        wordforms_tsvpath: str,
        max_given_constraints: int = 3,
        max_words_per_constraint: int = 3,
    ):
        self._vocabulary = vocabulary
        self._max_given_constraints = max_given_constraints
        self._max_words_per_constraint = max_words_per_constraint

        self._num_main_states = 2 ** max_given_constraints
        self._num_total_states = self._num_main_states * max_words_per_constraint

        self._wordforms: Dict[str, List[str]] = {}
        with open(wordforms_tsvpath, "r") as wordforms_file:
            reader = csv.DictReader(
                wordforms_file, delimiter="\t", fieldnames=["class_name", "words"]
            )
            for row in reader:
                self._wordforms[row["class_name"]] = row["words"].split(",")

    def build(self, constraints: List[str]):
        r"""
        Build a finite state machine given a list of constraints.

        Parameters
        ----------
        constraints: List[str]
            A list of up to three (possibly) multi-word constraints, in our use-case these are
            Open Images object class names.

        Returns
        -------
        Tuple[torch.Tensor, int]
            A finite state machine as an adjacency matrix, index of the next available unused
            sub-state. This is later used to trim the unused sub-states from FSM.
        """
        fsm = torch.zeros(self._num_total_states, self._num_total_states, dtype=torch.uint8)

        # Self loops for all words on main states.
        fsm[range(self._num_main_states), range(self._num_main_states)] = 1

        fsm = fsm.unsqueeze(-1).repeat(1, 1, self._vocabulary.get_vocab_size())

        substate_idx = self._num_main_states
        for i, constraint in enumerate(constraints):
            fsm, substate_idx = self._add_nth_constraint(fsm, i + 1, substate_idx, constraint)

        return fsm, substate_idx

    def _add_nth_constraint(self, fsm: torch.Tensor, n: int, substate_idx: int, constraint: str):
        r"""
        Given an (incomplete) FSM matrix with transitions for "(n - 1)" constraints added, add
        all transitions for the "n-th" constraint.

        Parameters
        ----------
        fsm: torch.Tensor
            A tensor of shape ``(num_total_states, num_total_states, vocab_size)`` representing an
            FSM under construction.
        n: int
            The cardinality of constraint to be added. Goes as 1, 2, 3... (not zero-indexed).
        substate_idx: int
            An index which points to the next unused position for a sub-state. It starts with
            ``(2 ** num_main_states)`` and increases according to the number of multi-word
            constraints added so far. The calling method, :meth:`build` keeps track of this.
        constraint: str
            A (possibly) multi-word constraint, in our use-case it is an Open Images object class
            name.

        Returns
        -------
        Tuple[torch.Tensor, int]
            FSM with added connections for the constraint and updated ``substate_idx`` pointing to
            the next unused sub-state.
        """
        words = constraint.split()
        connection_stride = 2 ** (n - 1)

        from_state = 0
        while from_state < self._num_main_states:
            for _ in range(connection_stride):
                word_from_state = from_state
                for i, word in enumerate(words):
                    # fmt: off
                    # Connect to a sub-state for all words in multi-word constraint except last.
                    if i != len(words) - 1:
                        fsm = self._connect(
                            fsm, word_from_state, substate_idx, word, reset_state=from_state
                        )
                        word_from_state = substate_idx
                        substate_idx += 1
                    else:
                        fsm = self._connect(
                            fsm, word_from_state, from_state + connection_stride, word,
                            reset_state=from_state,
                        )
                    # fmt: on
                from_state += 1
            from_state += connection_stride
        return fsm, substate_idx

    def _connect(
        self, fsm: torch.Tensor, from_state: int, to_state: int, word: str, reset_state: int = None
    ):
        r"""
        Add a connection between two states for a particular word (and all its word-forms). This
        means removing self-loop from ``from_state`` for all word-forms of ``word`` and connecting
        them to ``to_state``.
        
        Extended Summary
        ----------------
        In case of multi-word constraints, we return back to the ``reset_state`` for any utterance
        other than ``word``, to satisfy a multi-word constraint if all words are decoded
        consecutively. For example: for "fire hydrant" as a constraint between Q0 and Q1, we reach
        a sub-state "Q8" on decoding "fire". Go back to main state "Q1" on decoding "hydrant"
        immediately after, else we reset back to main state "Q0".

        Parameters
        ----------
        fsm: torch.Tensor
            A tensor of shape ``(num_total_states, num_total_states, vocab_size)`` representing an
            FSM under construction.
        from_state: int
            Origin state to make a state transition.
        to_state: int
            Destination state to make a state transition.
        word: str
            The word which serves as a constraint for transition between given two states.
        reset_state: int, optional (default = None)
           State to reset otherwise. This is only valid if ``from_state`` is a sub-state.

        Returns
        -------
        torch.Tensor
            FSM with the added connection.
        """
        wordforms = self._wordforms[word]
        wordform_indices = [self._vocabulary.get_token_index(w) for w in wordforms]

        for wordform_index in wordform_indices:
            fsm[from_state, to_state, wordform_index] = 1
            fsm[from_state, from_state, wordform_index] = 0

        if reset_state is not None:
            fsm[from_state, from_state, :] = 0
            fsm[from_state, reset_state, :] = 1
            for wordform_index in wordform_indices:
                fsm[from_state, reset_state, wordform_index] = 0

        return fsm
