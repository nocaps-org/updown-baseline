UpDown Captioner Baseline for ``nocaps``
========================================

Baseline model for `nocaps <https://nocaps.org>`_ benchmark, a re-implementation based on the
`UpDown image captioning model trained on the COCO dataset (only) <https://github.com/peteanderson80/up-down-captioner>`_
and with added support of decoding using `Constrained Beam Search <https://arxiv.org/abs/1612.00576>`_..

.. image:: _static/qualitative_examples.jpg


Get the source at `@nocaps-org/updown-baseline <https://www.github.com/nocaps-org/updown-baseline>`_.


If you find this code useful, please consider citing:

.. code-block:: text

    @inproceedings{nocaps2019,
    author    = {Harsh Agrawal* and Karan Desai* and Yufei Wang and Xinlei Chen and Rishabh Jain and
                Mark Johnson and Dhruv Batra and Devi Parikh and Stefan Lee and Peter Anderson},
    title     = {{nocaps}: {n}ovel {o}bject {c}aptioning {a}t {s}cale},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year      = {2019}
    }

As well as the paper that proposed this model:

.. code-block:: text

    @inproceedings{Anderson2017up-down,
        author    = {Peter Anderson and Xiaodong He and Chris Buehler and Damien Teney and Mark Johnson
                    and Stephen Gould and Lei Zhang},
        title     = {Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering},
        booktitle = {CVPR},
        year      = {2018}
    }


User Guide
==========

.. toctree::
    :maxdepth: 1

    setup_dependencies
    training
    evaluation_inference


API Reference
=============

.. toctree::
    :maxdepth: 2

    updown/config
    updown/data
    updown/models
    updown/modules
    updown/utils


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
