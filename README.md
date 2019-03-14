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

This codebase requires Python 3.6+ or higher. It uses PyTorch v1.0, and has out of the box support with CUDA 9 and CuDNN 7. The recommended way to set this codebase up is through Anaconda or Miniconda, although this should work just as fine with VirtualEnv.

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

[todo]


[1]: nocaps.org
[2]: https://kdexd.github.io/probnmn-clevr/probnmn/usage/setup_dependencies.html
[3]: https://kdexd.github.io/probnmn-clevr/probnmn/usage/training.html
[4]: https://kdexd.github.io/probnmn-clevr/probnmn/usage/evaluation_inference.html
