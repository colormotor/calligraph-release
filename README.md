# Calligraph
Code for the paper:
### Neural Image Abstraction Using Long Smoothing B-Splines

## Conda (recommended)

The ideal way to get this working is installing the conda/mamba package manager through miniforge. On Mac/Linux, from a terminal do

    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh

Say &ldquo;yes&rdquo; to everything during the installation process and restart the terminal
when done.

On Windows open [this link](https://github.com/conda-forge/miniforge) and download the first Windows, x86<sub>64</sub> installer that
appears in the lists. Again &ldquo;Say yes to everything&rdquo;. This will install a
&ldquo;Miniforge Prompt&rdquo; application that you can use to install dependencies.

It is recommended to create a new environment to install the dependencies, which can be done with

    mamba create -n yourenvname python=3.10

Then proceed with the following dependencies. In practice I think torch installs scipy/numpy/matplotlib, but it might be useful to install these first with

    mamba install numpy scipy matplotlib

making sure your environment is active.


## Dependencies

-   Install NumPy, SciPy, matplotlib, OpenCV
-   Install [torch/torchvision](https://pytorch.org/get-started/locally/) following your system specs
-   Install DiffVg from the [colormotor branch](https://github.com/colormotor/diffvg) (has thick strokes fix)
    -   Clone the Diffvg repo then from the directory do:
        -   `git submodule update --init --recursive` and then
        -   `python setup.py install`
-   Install ftfy `mamba install ftfy`


<a id="org30cfaee"></a>

### Optional

-   Install [geomloss](https://www.kernel-operations.io/geomloss/) to test Geometric Loss (Sinkhorn)


<a id="org5c2e1d9"></a>

## Install

Finally, install locally from the repo directory with

    pip install -e .


<a id="org305a34e"></a>

# Examples
