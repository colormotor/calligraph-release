# Calligraph
Code for the paper:
### Neural Image Abstraction Using Long Smoothing B-Splines

## Conda (recommended)

The ideal way to get this working is installing the conda/mamba package manager through miniforge. On Mac/Linux, from a terminal do

    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh

It is recommended to create a new environment to install the dependencies, which can be done with

    mamba create -n yourenvname python=3.10

Then proceed with the following dependencies. In practice I think torch installs scipy/numpy/matplotlib, but it might be useful to install these first with

    mamba install numpy scipy matplotlib opencv scikit-image

making sure your environment is active.

## Dependencies
-   Install NumPy, SciPy, matplotlib, OpenCV (using mamba as above or pip)
-   Install [torch/torchvision](https://pytorch.org/get-started/locally/)
    following your system specs
-   Install DiffVg from the [colormotor branch](https://github.com/colormotor/diffvg) (has thick strokes fix):
    -   clone the repo: `git clone https://github.com/colormotor/diffvg.git`
    -   From the repo directory do:
        -   `git submodule update --init --recursive` and then
        -   `python setup.py install`
-   Install remaining deps with pip:
    - `pip install accelerate transformers diffusers ortools open-clip-torch`


## Install locally

Finally, install locally from the repo directory with

    pip install -e .


# Examples
