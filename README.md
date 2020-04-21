[![image](https://travis-ci.com/ComPWA/tensorwaves.svg?branch=master)](https://travis-ci.com/ComPWA/tensorwaves)
[![image](https://codecov.io/gh/ComPWA/tensorwaves/branch/master/graph/badge.svg)](https://codecov.io/gh/ComPWA/tensorwaves)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/db8f89e5588041d8a995968262c224ef)](https://www.codacy.com/gh/ComPWA/tensorwaves?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ComPWA/tensorwaves&amp;utm_campaign=Badge_Grade)

# TensorWaves

*This package is Work-In-Progress and currently instable.*

TensorWaves is a Python package for doing Partial Wave Analysis with [TensorFlow](https://www.tensorflow.org/) as computational backend. The package originates from [pycompwa](https://compwa.github.io/) (see in particular the [expert system](./tensorwaves/expertsystem)), which did its computations through ([ComPWA](https://github.com/ComPWA/ComPWA)) ― ComPWA's function tree is similar to TensorFlow graphs and can therefore be replaced by the tools that TensorFlow already offers. The decision to completely migrate ComPWA to TensorFlow has been further inspired by [TensorFlowAnalysis](https://gitlab.cern.ch/poluekt/TensorFlowAnalysis) and [zfit](https://github.com/zfit/zfit/).


## Goals we strive for

Our aim is simple: facilitate doing partial wave analysis with TensorFlow. Since partial wave analysis requires knowledge from many PWA experts, the **developer experience** has our highest priority. For this reason, we try to make as much use of the fact that Python is an easy and flexible language and that TensorFlow is continuously simplifying its interfaces. The following rules of thumb may be of help:

- It should be **straightforward to find and implement new formulas**, so class hierarchies should be only introduced once necessary.

- **Follow a clean design**, so that it's easy to find one's way around as a physicist. Try to reduce dependencies between modules and categorise submodules into main modules.

These ideas resonate with the [Zen of Python](https://www.python.org/dev/peps/pep-0020/): keep it simple, keep it easy to contribute. Physics research is our end goal after all.


## Installation

As for now this repository is purely developmental (see [Contribute](#contribute)), so there is no install procedure. Instead, it is easiest to work in a virtual Python environment and have this repository added as a development directory. In that sense, the instructions here are useful in general if you want to experiment with Python development! Now, after cloning this repository and moving into its main directory, you can opt to use either Conda or PyPI.

### Option 1: Conda
The easiest way to work with these packages is to use [Anaconda/Conda](https://www.anaconda.com/). This allows you to contain all dependencies you need for this project within a virtual environment that you can easily throw away or replace later on if you run into trouble. You can download Conda [here](https://www.anaconda.com/distribution/#download-section). It can be installed without admin rights on any system!

Next steps are:
1. Add [Conda-Forge](https://conda-forge.org/) as a channel to your Conda installation:
   ```bash
   conda config --add channels conda-forge
   conda config --set channel_priority strict
   ```

2. Create a Conda environment named `tensorwaves` (or whatever you want) and initialize it with the necessary packages. The required dependencies are listed in the [`requirements.txt`](./requirements.txt) file, apart from Python itself:
   ```bash
   conda create --name tensorwaves python --file requirements.txt
   ```

3. Activate the environment using:
   ```bash
   conda activate tensorwaves
   ```
   You can see that you are in a 'virtual environment' by typing `which python` and/or `python --version`—you are now using a Python interpreter of the environment.

4. Now the most important step! Activate the main directory of the repository as a Conda ['development mode'](https://docs.conda.io/projects/conda-build/en/latest/resources/commands/conda-develop.html) directory by running
   ```bash
   conda develop .
   ```
   **from the main directory**. This means that the `tensorwaves` module located within this folder becomes available in the Python interpreter (and Jupyter notebook), so you can can then just run `import tensorwaves` from any other directory on your system.

Note that you can easily switch back with `conda deactivate`. And if you want to trash this environment to start all over if you messed up something, just run `conda remove --name tensorwaves --all`.

### Option 2: Python Package Index
If Conda is not available on your system, you can go the conventional route: using [PyPI](https://pypi.org/) (`pip`). In order to make the TensorWaves packages known to your Python interpreter, you will have to use [`virtualenvwrapper`](https://virtualenvwrapper.readthedocs.io/en/latest/) (just like `conda develop`). Again, it is safest if you do this by working in a virtual environment. So before you get going, make sure you have Python3's [`venv`](https://docs.python.org/3/library/venv.html) installed on your system.

Now, let's go:
1. Create a virtual environment (and call it `venv`):
   ```bash
   python3 -m venv ./venv
   ```
   Note that we append `python3 -m` to ensure that we use the `venv` module of Python3.

2. Activate the virtual environment:
   ```bash
   source ./venv/bin/activate
   ```
   If this went correctly, you should see `(venv)` on your command line and `which python3` should point to `venv/bin`.

3. Set the directory of this repository as a development path. For this you need to install and activate `virtualenvwrapper`, then you can use the command `add2virtualenv`:
   ```bash
   python3 -m pip install virtualenvwrapper
   source venv/bin/virtualenvwrapper.shF
   add2virtualenv .
   ```
   where we assume you run `add2virtualenv` from the TensorWaves directory. You can use `add2virtualenv -d .` to unregister the path again.

Now, as with Conda, the nice thing is that if you run into trouble with conflicting packages or so, just trash the `venv` folder and start over!


## Documentation

An API of the TensorWaves package can be generated locally. If you have followed the [installation instructions](#installation), just do the following:

1. Navigate into the [`doc`](./doc) folder.
2. Install the required packages (either `conda install --file requirements.txt` or `pip install -r requirements.txt`).
3. Run `make html` and/or `make pdflatex`.
4. See the output in the `_build` folder!


## Contribute

See [`CONTRIBUTING.md`](./CONTRIBUTING.md)
