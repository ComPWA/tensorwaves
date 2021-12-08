# Installation

[![PyPI package](https://badge.fury.io/py/tensorwaves.svg)](https://pypi.org/project/tensorwaves)
[![Conda package](https://anaconda.org/conda-forge/tensorwaves/badges/version.svg)](https://anaconda.org/conda-forge/tensorwaves)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/tensorwaves)](https://pypi.org/project/tensorwaves)

The fastest way of installing this package is through PyPI or Conda:

:::{tabbed} PyPI

```shell
python3 -m pip install tensorwaves
```

:::

:::{tabbed} Conda

```shell
conda install -c conda-forge tensorwaves
```

:::

This installs the
[latest, stable release](https://github.com/ComPWA/tensorwaves/releases) that
you can find on the
[`stable`](https://github.com/ComPWA/tensorwaves/tree/stable) branch.

Optional dependencies can be installed as follows:

```shell
pip install tensorwaves[pwa]  # installs tensorwaves with ampform
pip install tensorwaves[jax,scipy]
pip install tensorwaves[all]  # all runtime dependencies
```

::::{dropdown} **GPU support**

<!-- cspell:ignore cudnn dpkg jaxlib nvcc -->

Computations with are fastest on a
[GPU](https://en.wikipedia.org/wiki/Graphics_processing_unit). To get JAX and
TensorFlow work with your graphics card, you need to install at least
[CUDA Toolkit **11.2**](https://developer.nvidia.com/cuda-downloads) and
[cuDNN **8.1**](https://developer.nvidia.com/cudnn).

Below is an installation guide for installing TF and JAX with GPU support.
These instructions may become outdated, so refer to
[this TF page](https://www.tensorflow.org/install/gpu) an
[these JAX instructions](https://github.com/google/jax#pip-installation-gpu-cuda)
if you run into trouble.

1. Download and install CUDA Toolkit **11.x** by following
   [these platform-dependent instructions](https://developer.nvidia.com/cuda-downloads).
   There may be dependency conflicts with existing NVIDIA packages. In that
   case, have a look
   [here](https://forums.developer.nvidia.com/t/cuda-install-unmet-dependencies-cuda-depends-cuda-10-0-10-0-130-but-it-is-not-going-to-be-installed/66488/6?u=user85126).
2. Tell the system where to find CUDA. In Ubuntu (or WSL-Ubuntu), this can be
   done by adding the following line to your
   [`.bashrc`](https://unix.stackexchange.com/a/129144) file:

   ```shell
   export PATH="$PATH:/usr/local/cuda/bin"
   ```

   But first, check if `/usr/local/cuda/bin` indeed exists!

3. Restart your machine. Then, launch a terminal to
   [check whether CUDA is installed](https://stackoverflow.com/a/9730706):

   ```shell
   nvcc --version
   ```

4. Download cuDNN from [this page](https://developer.nvidia.com/cudnn). You
   need to create an NVIDIA account for this first. Make sure that you download
   cuDNN **for CUDA 11.x**!
5. Install cuDNN following
   [these instructions](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).
   In Ubuntu, this can be done with:

   ```shell
   sudo dpkg -i cudnn-local-repo-ubuntu*.deb
   ```

6. **[Recommended]** Install JAX with GPU binaries in your
   {ref}`virtual environment <compwa-org:develop:Virtual environment>`
   following
   [these instructions](https://github.com/google/jax#pip-installation-gpu-cuda).

   ```shell
   pip install --upgrade jax[cuda] jaxlib -f https://storage.googleapis.com/jax-releases/jax_releases.html
   ```

   It doesn't matter whether you do this before or after installing
   TensorWaves.

7. If TensorFlow can correctly find your GPU, the following should return a
   non-empty list:

   ```shell
   python3 -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'

   ```

   If JAX can correctly find your GPU, the following should return `gpu`:

   ```shell
   python3 -c 'from jax.lib import xla_bridge; print(xla_bridge.get_backend().platform)'
   ```

::::

The latest version on the
[`main`](https://github.com/ComPWA/tensorwaves/tree/main) branch can be
installed as follows:

```shell
python3 -m pip install git+https://github.com/ComPWA/tensorwaves@main
```

In that case, however, we highly recommend using the more dynamic
{ref}`'editable installation' <compwa-org:develop:Editable installation>`
instead. This goes as follows:

1. Get the source code:

   ```shell
   git clone https://github.com/ComPWA/tensorwaves.git
   cd tensorwaves
   ```

2. **[Recommended]** Create a virtual environment (see
   {ref}`here <compwa-org:develop:Virtual environment>`).

3. Install the project as an
   {ref}`'editable installation' <compwa-org:develop:Editable installation>`
   and install
   {ref}`additional packages <compwa-org:develop:Optional dependencies>` for
   the developer:

   ```shell
   python3 -m pip install -e .[dev]
   ```

   :::{dropdown} Pinning dependency versions

   In order to install the _exact same versions_ of the dependencies with which
   the framework has been tested, use the provided
   [constraints files](https://pip.pypa.io/en/stable/user_guide/#constraints-files)
   for the specific Python version `3.x` you are using:

   ```shell
   python3 -m pip install -c .constraints/py3.x.txt -e .[dev]
   ```

   ```{seealso}

   {ref}`develop:Pinning dependency versions`

   ```

   :::

That's all! Have a look at the {doc}`/usage` page to try out the package. You
can also have a look at the {doc}`compwa-org:develop` page for tips on how to
work with this 'editable' developer setup!
