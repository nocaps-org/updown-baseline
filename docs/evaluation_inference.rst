How to evaluate or do inference?
================================

Generate predictions for ``nocaps`` val or ``nocaps`` test using a pretrained checkpoint as
follows.

.. code-block::

    python scripts/inference.py \
        --config /path/to/config.yaml \
        --checkpoint-path /path/to/checkpoint.pth \
        --output-path /path/to/save/predictions.json \
        --gpu-ids 0

Add ``--evalai-submit`` flag if you wish to submit the predictions directly to EvalAI and get
results.


Using Constrained Beam Search
-----------------------------

To use Constrained Beam Search during inference, add `--config-override MODEL.USE_CBS True`
or specify it in the config file. This is only valid for checkpoints trained with frozen
GloVe embeddings (``MODEL.EMBEDDING_SIZE 300``).

