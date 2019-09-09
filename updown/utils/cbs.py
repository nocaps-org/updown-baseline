import csv

from allennlp.data import Vocabulary
import torch

from torchtext.vocab import GloVe


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


def initialize_glove(
    vocabulary: Vocabulary, namespace: str = "tokens"
) -> torch.Tensor:
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
