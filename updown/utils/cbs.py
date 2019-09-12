import csv
import json

import anytree
from anytree.search import findall
import numpy as np
import torch
from torchtext.vocab import GloVe
from allennlp.data import Vocabulary


def add_constraint_words_to_vocabulary(
    vocabulary: Vocabulary, constraint_words_filepath: str, namespace: str = "tokens"
) -> Vocabulary:
    r"""
    Expand the :class:`~allennlp.data.vocabulary.Vocabulary` with CBS constraint words. We do not
    need to worry about duplicate words in constraints and caption vocabulary. AllenNLP avoids
    duplicates automatically.

    Parameters
    ----------
    vocabulary: allennlp.data.vocabulary.Vocabulary
        The vocabulary to be expanded with provided words.
    constraint_words_filepath: str
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

    with open(constraint_words_filepath, "r") as constraint_words_file:
        reader = csv.DictReader(
            constraint_words_file, delimiter="\t", fieldnames=["class", "words"]
        )
        for row in reader:
            for word in row["words"].split(","):
                # Constraint words can be "multi-word" (may have more than one tokens).
                # Add all tokens to the vocabulary separately.
                for w in word.split():
                    vocabulary.add_token_to_namespace(w, namespace)

    return vocabulary


def initialize_glove(vocabulary: Vocabulary, namespace: str = "tokens") -> torch.Tensor:
    r"""
    Initialize embeddings of all the tokens in a given
    :class:`~allennlp.data.vocabulary.Vocabulary` by their GloVe vectors.

    Extended Summary
    ----------------
    It is recommended to train an :class:`~updown.models.updown_captioner.UpDownCaptioner` with
    frozen word embeddings when one wishes to perform Constrained Beam Search decoding during
    inference. This is because the constraint words may not appear in caption vocabulary (out of
    domain), and their embeddings will never be updated during training. Initializing with frozen
    GloVe embeddings is helpful, because they capture more meaningful semantics than randomly
    initialized embeddings.

    Parameters
    ----------
    vocabulary: allennlp.data.vocabulary.Vocabulary
        The vocabulary containing tokens to be initialized.
    namespace: str, optional (default="tokens")
        The namespace of :class:`~allennlp.data.vocabulary.Vocabulary` to add these words.

    Returns
    -------
    Use this tensor to initialize :class:`~torch.nn.Embedding` layer in
    :class:~updown.models.updown_captioner.UpDownCaptioner` as:

    >>> embedding_layer = torch.nn.Embeddingfrom_pretrained(weights, freeze=True)
    """
    glove = GloVe(name="42B", dim=300)

    caption_oov = 0

    glove_vectors = torch.zeros(vocabulary.get_vocab_size(), 300)
    for word, i in vocabulary.get_token_to_index_vocabulary().items():

        # Words not in GloVe vocablary would be initialized as zero vectors.
        if word in glove.stoi:
            glove_vectors[i] = glove.vectors[glove.stoi[word]]
        else:
            caption_oov += 1

    print(f"{caption_oov} out of {glove_vectors.size(0)} words were not found in GloVe.")
    return glove_vectors


def read_hierarchy(hierarchy_jsonpath: str) -> anytree.AnyNode:
    def __import(data, parent=None):
        assert isinstance(data, dict)
        assert "parent" not in data
        attrs = dict(data)
        children = attrs.pop("Subcategory", []) + attrs.pop("Part", [])
        node = AnyNode(parent=parent, **attrs)
        for child in children:
            __import(child, parent=node)
        return node

    data = json.load(open(hierarchy_jsonpath))
    return __import(data)


def nms(boxes, classes, hierarchy, thresh=0.7):
    # Non-max suppression of overlapping boxes where score is based on 'height' in the hierarchy,
    # defined as the number of edges on the longest path to a leaf

    scores = [
        findall(hierarchy, filter_=lambda node: node.LabelName in (object_class))[0].height
        for object_class in classes
    ]

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

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
