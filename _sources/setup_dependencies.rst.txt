How to setup this codebase?
===========================

This codebase requires Python 3.6 or higher. The recommended way to set up this
codebase up through Anaconda/Miniconda.

Install Dependencies
--------------------

1. Install Anaconda or Miniconda distribution based on Python3+ from their
   `downloads site <https://conda.io/docs/user-guide/install/download.html>`_.

2. Clone the repository.

    .. code-block:: shell

        git clone https://www.github.com/nocaps-org/updown-baseline
        cd updown-baseline

3. Create a conda environment and install all the dependencies, and this
   codebase as a package in development version.

    .. code-block:: shell

        conda create -n updown python=3.6
        conda activate updown
        pip install -r requirements.txt
        python setup.py develop

    .. note::

        If ``evalai`` package install fails, install ``libxml2-dev`` and
        ``libxstl1-dev`` via ``apt``.

Now you can ``import updown`` from anywhere in your filesystem as long as you have this conda
environment activated.


Download Image Features
-----------------------

We provide pre-extracted bottom-up features for COCO and ``nocaps`` splits. These are extracted
using a Faster-RCNN detector pretrained on Visual Genome, made available by
`Anderson et al. 2017 <https://arxiv.org/abs/1707.07998>`_. We call this ``VG Detector``.
We extract features from 100 region proposals for an image, and select them based on a confidence
threshold of 0.2 - we finally get 10-100 features per image (adaptive). 

Download (or symlink) the image features under ``$PROJECT_ROOT/data`` directory:

  - coco_train2017_vg_detector_features_adaptive.h5_
  - coco_val2017_vg_detector_features_adaptive.h5_
  - nocaps_val_vg_detector_features_adaptive.h5_
  - nocaps_test_vg_detector_features_adaptive.h5_

.. seealso::

    Our image-feature-extractors_ repo for more info on ``VG Detector``, and how these
    features are extracted from it.


Download Annotation Files
-------------------------

Download COCO Captions and `nocaps` val/test image info and arrange in a directory structure as
follows:

.. code-block::

    $PROJECT_ROOT/data
        |-- coco
        |   +-- annotations
        |       |-- captions_train2017.json
        |       +-- captions_val2017.json
        +-- nocaps
            +-- annotations
                |-- nocaps_val_image_info.json
                +-- nocaps_test_image_info.json

- COCO Captions: http://images.cocodataset.org/annotations/annotations_trainval2017.zip  
- nocaps val image info: https://s3.amazonaws.com/nocaps/nocaps_val_image_info.json  
- nocaps test image info: https://s3.amazonaws.com/nocaps/nocaps_test_image_info.json  



[Optional] Download files for Constrained Beam Search
-----------------------------------------------------

1. If you wish to decode using `Constrained Beam Search <https://arxiv.org/abs/1612.00576>`_,
   download pre-extracted detections from a detector trained using Open Images (we call it
   ``OI Detector`) into ``$PROJECT_ROOT/data``.

  - nocaps_val_oi_detector_boxes.json_ (in COCO bounding box annotations format)
  - nocaps_test_oi_detector_boxes.json_ (in COCO bounding box annotations format)

2. Download Open Images meta data files into ``$PROJECT_ROOT/data/cbs``:

  - class_hierarchy.json_ : A hierarchy of object classes
    `declared by Open Images <https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy_visualizer/circle.html>`_.
    Our file is in a format which is more human-readable.
  - constraint_wordforms.tsv_ : wordforms of all words which could be CBS constraints.
    This is how one could allow either of singular-plural words to satisfy a constraint (or even
    close synonym words).


.. seealso::

    Our image-feature-extractors_ repo for more info on ``OI Detector``, and how these
    bounding box detections are extracted from it.


Build Vocabulary
----------------

Build caption vocabulary using COCO train2017 captions.

.. code-block::

    python scripts/build_vocabulary.py -c data/coco/captions_train2017.json -o data/vocabulary


Evaluation Server
-----------------

``nocaps`` val and test splits are held privately behind EvalAI. To evaluate on ``nocaps``,
create an account on EvalAI_ and get the auth token from
`profile details <http://evalai.cloudcv.org/web/profile>`_. Set the token through EvalAI CLI:

.. code-block::

    evalai set_token <your_token_here>


You are all set to use this codebase!
-------------------------------------


.. _image-feature-extractors: https://github.com/nocaps-org/image-feature-extractors

.. _coco_train2017_vg_detector_features_adaptive.h5: https://www.dropbox.com/s/n0lj4oandy71sl8/coco_train2017_vg_detector_features_adaptive.h5
.. _coco_val2017_vg_detector_features_adaptive.h5: https://www.dropbox.com/s/tzldnj7xwxxcnp6/coco_val2017_vg_detector_features_adaptive.h5
.. _nocaps_val_vg_detector_features_adaptive.h5: https://www.dropbox.com/s/6qqebcybfebrloe/nocaps_val_vg_detector_features_adaptive.h5
.. _nocaps_test_vg_detector_features_adaptive.h5: https://www.dropbox.com/s/tl3sdfdgpbafs2c/nocaps_test_vg_detector_features_adaptive.h5

.. _nocaps_val_oi_detector_boxes.json: https://www.dropbox.com/s/ro6c4acnf5snnr5/nocaps_val_oi_detector_boxes.json
.. _nocaps_test_oi_detector_boxes.json: https://www.dropbox.com/s/s4a7u0u1he3uh4j/nocaps_test_oi_detector_boxes.json
.. _class_hierarchy.json: https://www.dropbox.com/s/0i6sxy400tb0scp/class_hierarchy.json
.. _constraint_wordforms.tsv: https://www.dropbox.com/s/6a36qw7ryi4dygb/constraint_wordforms.tsv

.. _EvalAI: http://evalai.cloudcv.org