How to train your captioner?
============================

We manage experiments through config files -- a config file should contain arguments which are
specific to a particular experiment, such as those defining model architecture, or optimization
hyperparameters. Other arguments such as GPU ids, or number of CPU workers should be declared in
the script and passed in as argparse-style arguments.

UpDown Captioner (without CBS)
------------------------------

Train a baseline UpDown Captioner with all the default hyperparameters as follows. This would
reproduce results of the first row in ``nocaps val`` table from our paper.

.. code-block::

    python scripts/train.py \
        --config configs/train_updown_coco_train2017.yaml \
        --gpu-ids 0 --serialization-dir checkpoints/updown
    
Refer :class:`updown.config.Config` for default hyperparameters.
For other configurations, write your own config file, and/or a set of key-value pairs through
``--config-override`` argument. For example:

.. code-block::

    python scripts/train.py \
        --config configs/train_updown_coco_train2017.yaml \
        --config-override OPTIM.BATCH_SIZE 250 \
        --gpu-ids 0 --serialization-dir checkpoints/updown-baseline

.. note::

    This configuration uses randomly initialized word embeddings, which are trained during
    training. It is not possible to run Constrained Beam Search on this checkpoint.


UpDown Captioner (with CBS)
---------------------------

Train a baseline UpDown Captioner with Constrained Beam Search decoding during evaluation. This
would reproduce results of the second row in ``nocaps val`` table from our paper.

.. code-block::

    python scripts/train.py \
        --config configs/train_updown_plus_cbs_coco_train2017.yaml \
        --gpu-ids 0 --serialization-dir checkpoints/updown_plus_cbs


The only difference with original config is the word embedding size, this one is set to the
GloVe dimension (300), and frozen during training. A checkpoint trained using this config can
be run without Constrained Beam Search decoding.


Additional Details
------------------

Multi-GPU Training
******************

Multi-GPU training is fully supported, pass GPU IDs as ``--gpu-ids 0 1 2 3``.

Saving Model Checkpoints
************************

This script serializes model checkpoints every few iterations, and keeps track of best performing
checkpoint based on overall CIDEr score. 

.. seealso::

    :class:`updown.utils.checkpointing.CheckpointManager` for more detail on how checkpointing is
    managed. A copy of configuration file used for a particular experiment is also saved under
    ``--serialization-dir``.

Logging
*******

This script logs loss curves and metrics to Tensorboard, log files are at ``--serialization-dir``.
Execute ``tensorboard --logdir /path/to/serialization_dir --port 8008`` and visit
``localhost:8008`` in the browser.
