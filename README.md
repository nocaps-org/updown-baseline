UpDown Captioner Baseline for `nocaps`
=====================================

Baseline model for [`nocaps`][1] dataset.

**[nocaps: novel object captioning at scale][1]**  
Harsh Agrawal*, Karan Desai*, Yufei Wang, Xinlei Chen, Rishabh Jain,  
Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, Peter Anderson

If you find this code useful, please consider citing:

```text
@Article{nocaps,
    Author  = {Harsh Agrawal* and Karan Desai* and Yufei Wang and Xinlei Chen and Rishabh Jain and Mark Johnson and Dhruv Batra and Devi Parikh and Stefan Lee and Peter Anderson},
    Title   = {{nocaps}: {n}ovel {o}bject {c}aptioning {a}t {s}cale},
    Journal = {arXiv preprint arXiv:1812.08658},
    Year    = {2018},
}
```


How to setup ths codebase?
--------------------------

This codebase requires Python 3.6+ or higher. It uses PyTorch v1.1, and has out of the box support with CUDA 9 and CuDNN 7. The recommended way to set this codebase up is through Anaconda or Miniconda, although this should work just as fine with VirtualEnv.

### Install Dependencies

1. Install Anaconda or Miniconda distribution based on Python3+ from their [downloads' site][2].

1. Clone the repository first.

    ```
    git clone https://www.github.com/nocaps-org/updown-baseline
    cd updown-baseline
    ```

1. Create a conda environment and install all the dependencies.

    ```
    conda create -n updown python=3.6
    conda activate updown
    pip install -r requirements.txt
    ```

1. Install this codebase as a package in development version.

    ```
    python setup.py develop
    ```

Now you can `import updown` from anywhere in your filesystem as long as you have this conda environment activated.


### Download Data

#### Image Features

We provide bottom-up image features for COCO train2017, COCO val2017, nocaps val and nocaps test splits extracted using Faster-RCNN pre-trained on Visual Genome. Download the image features under `$PROJECT_ROOT/data` (or symlink them):

1. COCO train2017 (top-36 features): [link TODO]  
2. COCO val2017 (top-36 features): [link TODO]  
3. nocaps val (top-36 features): [link TODO]  
4. nocaps test (top-36 features): [link TODO]  

#### Annotations

Download COCO captions and nocaps val/test image info and arrange in a directory structure:

    ```
    $PROJECT_ROOT/data
        |-- coco
        |   |-- captions_train2017.json
        |   +-- captions_cal2017.json
        +-- nocaps
            |-- nocaps_val_image_info.json
            +-- nocaps_test_image_info.json
    ```

1. COCO captions: http://images.cocodataset.org/annotations/annotations_trainval2017.zip  
2. nocaps val image info: https://s3.amazonaws.com/nocaps/nocaps_val_image_info.json  
3. nocaps test image info: https://s3.amazonaws.com/nocaps/nocaps_test_image_info.json  


### Build vocabulary

Build caption vocabulary using COCO train2017 captions.

    ```
    python scripts/build_vocabulary.py -c data/coco/captions_train2017.json -o data/vocabulary
    ```


### EvalAI token

This codebase using EvalAI for evaluating on nocaps val and test splits. Create an account on EvalAI and get Auth token (from profile details).

    ```
    evalai set_token <your token here>
    ```

(**todo: screenshot**)


Training
--------

We provide a training script which accepts arguments as config files. The config file should contain arguments which are specific to a particular experiment, such as those defining model architecture, or optimization hyperparameters. Other arguments such as GPU ids, or number of CPU workers should be declared in the script and passed in as argparse-style arguments.

Train the baseline model provided in this repository as:

    ```
    python scripts/train.py --config configs/updown_nocaps_val.yml --gpu-ids 0 --serialization-dir checkpoints/updown_baseline
    ```

### Saving model checkpoints

This script will serialize model checkpoints every few iterations, and keep track of bestperforming checkpoint based on overall CIDEr score. Refer [updown/utils/checkpointing.py][2] for more details on how checkpointing is managed.

### Logging

Execute `tensorboard --logdir /path/to/serialization_dir --port 8008` and visit `localhost:8008` in the browser.


[1]: nocaps.org
[2]: https://github.com/kdex/updown-baseline/blob/master/updown/utils/checkpointing.py
[3]: https://www.github.com/lanpa/tensorboardX
