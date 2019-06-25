r"""This module provides package-wide configuration management."""
from typing import Any, List

from yacs.config import CfgNode as CN


class Config(object):
    r"""
    A collection of all the required configuration parameters. This class is a nested dict-like
    structure, with nested keys accessible as attributes. It contains sensible default values for
    all the parameters, which may be overriden by (first) through a YAML file and (second) through
    a list of attributes and values.

    Extended Summary
    ----------------
    This class definition contains default hyperparameters for the UpDown baseline from our paper.
    Modification of any parameter after instantiating this class is not possible, so you must
    override required parameter values in either through ``config_yaml`` or ``config_override``.

    Parameters
    ----------
    config_yaml: str
        Path to a YAML file containing configuration parameters to override.
    config_override: List[Any], optional (default= [])
        A list of sequential attributes and values of parameters to override. This happens after
        overriding from YAML file.

    Examples
    --------
    Let a YAML file named "config.yaml" specify these parameters to override::

        RANDOM_SEED: 42
        OPTIM:
          BATCH_SIZE: 512

    >>> _C = Config("config.yaml", ["OPTIM.BATCH_SIZE", 2048])
    >>> _C.RANDOM_SEED  # default: 0
    42
    >>> _C.OPTIM.BATCH_SIZE  # default: 150
    2048

    Attributes
    ----------
    RANDOM_SEED: 0
        Random seed for NumPy and PyTorch, important for reproducibility.
    __________

    DATA:
        Collection of required data paths for training and evaluation. All these are assumed to
        be relative to project root directory. If elsewhere, symlinking is recommended.

    DATA.VOCABULARY: "data/vocabulary"
        Path to a directory containing caption vocabulary (readable by AllenNLP).

    DATA.TRAIN_FEATURES: "data/coco_train2017_resnet101_faster_rcnn_genome_adaptive.h5"
        Path to an H5 file containing pre-extracted features from COCO train2017 images.

    DATA.VAL_FEATURES: "data/nocaps_val_resnet101_faster_rcnn_genome_adaptive.h5"
        Path to an H5 file containing pre-extracted features from nocaps val images.

    DATA.TEST_FEATURES: "data/nocaps_test_resnet101_faster_rcnn_genome_adaptive.h5"
        Path to an H5 file containing pre-extracted features from nocaps test images.

    DATA.TRAIN_CAPTIONS: "data/coco/annotations/captions_train2017.json"
        Path to a JSON file containing COCO train2017 captions in COCO format.

    DATA.VAL_CAPTIONS: "data/nocaps/annotations/nocaps_val_image_info.json"
        Path to a JSON file containing nocaps val image info.
        Captions are not available publicly.

    DATA.TEST_CAPTIONS: "data/nocaps/annotations/nocaps_test_image_info.json"
        Path to a JSON file containing nocaps test image info.
        Captions are not available publicly.

    DATA.MAX_CAPTION_LENGTH: 20
        Maximum length of caption sequences for language modeling. Captions longer than this will
        be truncated to maximum length.
    __________

    MODEL:
        Parameters controlling the model architecture of UpDown Captioner.

    MODEL.IMAGE_FEATURE_SIZE: 2048
        Size of the bottom-up image features.

    MODEL.EMBEDDING_SIZE: 1000
        Size of the word embedding input to the captioner.

    MODEL.HIDDEN_SIZE: 1200
        Size of the hidden / cell states of attention LSTM and language LSTM of the captioner.

    MODEL.ATTENTION_PROJECTION_SIZE: 768
        Size of the projected image and textual features before computing bottom-up top-down
        attention weights.

    MODEL.BEAM_SIZE: 5
        Beam size for finding the most likely caption during decoding time (evaluation).
    __________

    OPTIM:
        Optimization hyper-parameters, relevant during training a particular phase.

    OPTIM.BATCH_SIZE: 150
        Batch size during training and evaluation.

    OPTIM.NUM_ITERATIONS: 70000
        Number of iterations to train for, batches are randomly sampled.

    OPTIM.LR: 0.015
        Initial learning rate for SGD. This linearly decays to zero till the end of training.

    OPTIM.MOMENTUM: 0.9
        Momentum co-efficient for SGD.

    OPTIM.WEIGHT_DECAY: 0.001
        Weight decay co-efficient for SGD.

    OPTIM.CLIP_GRADIENTS
        Gradient clipping threshold to avoid exploding gradients.
    """

    def __init__(self, config_yaml: str, config_override: List[Any] = []):

        self._C = CN()
        self._C.RANDOM_SEED = 0

        self._C.DATA = CN()
        self._C.DATA.VOCABULARY = "data/vocabulary"

        self._C.DATA.TRAIN_FEATURES = "data/coco_train2017_resnet101_faster_rcnn_genome_adaptive.h5"
        self._C.DATA.VAL_FEATURES = "data/nocaps_val_resnet101_faster_rcnn_genome_adaptive.h5"
        self._C.DATA.TEST_FEATURES = "data/nocaps_test_resnet101_faster_rcnn_genome_adaptive.h5"

        self._C.DATA.TRAIN_CAPTIONS = "data/coco/captions_train2017.json"

        # These really don't contain the captions, just the image info.
        self._C.DATA.VAL_CAPTIONS = "data/nocaps/nocaps_val_image_info.json"
        self._C.DATA.TEST_CAPTIONS = "data/nocaps/nocaps_test_image_info.json"

        self._C.DATA.MAX_CAPTION_LENGTH = 20

        self._C.MODEL = CN()
        self._C.MODEL.IMAGE_FEATURE_SIZE = 2048
        self._C.MODEL.EMBEDDING_SIZE = 1000
        self._C.MODEL.HIDDEN_SIZE = 1200
        self._C.MODEL.ATTENTION_PROJECTION_SIZE = 768
        self._C.MODEL.BEAM_SIZE = 5

        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 150
        self._C.OPTIM.NUM_ITERATIONS = 70000
        self._C.OPTIM.LR = 0.015
        self._C.OPTIM.MOMENTUM = 0.9
        self._C.OPTIM.WEIGHT_DECAY = 0.001
        self._C.OPTIM.CLIP_GRADIENTS = 12.5

        # Override parameter values from YAML file first, then from override list.
        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

        # Make an instantiated object of this class immutable.
        self._C.freeze()

    def dump(self, file_path: str):
        r"""Save config at the specified file path.

        Parameters
        ----------
        file_path: str
            (YAML) path to save config at.
        """
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __str__(self):
        common_string: str = str(CN({"RANDOM_SEED": self._C.RANDOM_SEED})) + "\n"
        common_string += str(CN({"DATA": self._C.DATA})) + "\n"
        common_string += str(CN({"MODEL": self._C.MODEL})) + "\n"
        common_string += str(CN({"OPTIM": self._C.OPTIM})) + "\n"

        return common_string

    def __repr__(self):
        return self._C.__repr__()
