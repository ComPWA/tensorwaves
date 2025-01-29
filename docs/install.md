# Installation

[![PyPI package](https://badge.fury.io/py/tensorwaves.svg)](https://pypi.org/project/tensorwaves)
[![Conda package](https://anaconda.org/conda-forge/tensorwaves/badges/version.svg)](https://anaconda.org/conda-forge/tensorwaves)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/tensorwaves)](https://pypi.org/project/tensorwaves)

## Quick installation

The fastest way of installing this package is through PyPI or Conda:

:::{margin}

TensorWaves can work with different computational backends. They are provided through
{ref}`optional dependencies <compwa:develop:Optional dependencies>`.
[JAX](https://jax.readthedocs.io) is usually the fastest backend, so it's recommended to
install that as in the install examples here.

:::

::::{tab-set}
:::{tab-item} PyPI

```shell
python3 -m pip install tensorwaves[jax]
```

To install the package with support for {doc}`amplitude analysis </amplitude-analysis>`
run:

```shell
python3 -m pip install tensorwaves[jax,pwa]
```

:::
:::{tab-item} Conda

```shell
conda install -c conda-forge tensorwaves jax jaxlib
```

To install the package with support for {doc}`amplitude analysis </amplitude-analysis>`
run:

```shell
conda install -c conda-forge ampform phasespace
```

:::
::::

This installs the [latest release](https://github.com/ComPWA/tensorwaves/releases) that
you can find on the [`stable`](https://github.com/ComPWA/tensorwaves/tree/stable)
branch.

Optional dependencies can be installed as follows:

```shell
pip install tensorwaves[pwa]  # installs tensorwaves with ampform
pip install tensorwaves[jax,scipy,tf]
pip install tensorwaves[all]  # all runtime dependencies
```

:::::{container} full-width

::::{dropdown} **GPU support**

<!-- cspell:ignore cudnn dpkg jaxlib nvcc -->

Computations with are fastest on a
[GPU](https://en.wikipedia.org/wiki/Graphics_processing_unit). To get JAX and TensorFlow
work with your graphics card, you need to install at least
[CUDA Toolkit **11.2**](https://developer.nvidia.com/cuda-downloads) and
[cuDNN **8.1**](https://developer.nvidia.com/cudnn).

Below is an installation guide for installing TF and JAX with GPU support. These
instructions may become outdated, so refer to
[this TF page](https://www.tensorflow.org/install/gpu) an
[these JAX instructions](https://github.com/google/jax#pip-installation-gpu-cuda) if you
run into trouble.

1. Download and install CUDA Toolkit **11.x** by following
   [these platform-dependent instructions](https://developer.nvidia.com/cuda-downloads).
   There may be dependency conflicts with existing NVIDIA packages. In that case, have a
   look
   [here](https://forums.developer.nvidia.com/t/cuda-install-unmet-dependencies-cuda-depends-cuda-10-0-10-0-130-but-it-is-not-going-to-be-installed/66488/6?u=user85126).
2. Tell the system where to find CUDA. In Ubuntu (or WSL-Ubuntu), this can be done by
   adding the following line to your
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

4. Download and install cuDNN following
   [these instructions](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/index.html).
   Make sure that you download cuDNN **for CUDA 11.x**!

   In Ubuntu (Debian), there are two convenient options: (1)
   [installing through `apt`](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html#package-manager-local-installation)
   or (2)
   [using a local installer](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html#package-manager-local-installation).

5. **[Recommended]** Install JAX with GPU binaries in your
   {ref}`virtual environment <compwa:develop:Virtual environment>` following
   [these instructions](https://github.com/google/jax#pip-installation-gpu-cuda).

   ```shell
   pip install --upgrade jax[cuda] jaxlib -f https://storage.googleapis.com/jax-releases/jax_releases.html
   ```

   It doesn't matter whether you do this before or after installing TensorWaves.

6. If TensorFlow can correctly find your GPU, the following should return a non-empty
   list:

   ```shell
   python3 -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'

   ```

   If JAX can correctly find your GPU, the following should return `gpu`:

   ```shell
   python3 -c 'from jax.lib import xla_bridge; print(xla_bridge.get_backend().platform)'
   ```

::::

:::::

The latest version on the [`main`](https://github.com/ComPWA/tensorwaves/tree/main)
branch (or any other branch, tag, or commit) can be installed as follows:

```shell
python3 -m pip install git+https://github.com/ComPWA/tensorwaves@main
```

Or, with optional dependencies:

```{code-block} shell
:class: full-width
python3 -m pip install "tensorwaves[jax,pwa] @ git+https://github.com/ComPWA/tensorwaves@main"
```

## Developer installation

:::{include} ../CONTRIBUTING.md
:start-after: **[compwa.github.io/develop](https://compwa.github.io/develop)**!
:end-before: If the repository provides a Tox configuration
:::

That's all! Have a look at {doc}`/usage` to try out the package. You can also have a look at {doc}`compwa:develop` for tips on how to work with this 'editable' developer setup!
