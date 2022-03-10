# TensorWaves

[![10.5281/zenodo.5526650](https://zenodo.org/badge/doi/10.5281/zenodo.5526650.svg)](https://doi.org/10.5281/zenodo.5526650)
[![GPLv3+ license](https://img.shields.io/badge/License-GPLv3+-blue.svg)](https://www.gnu.org/licenses/gpl-3.0-standalone.html)

[![PyPI package](https://badge.fury.io/py/tensorwaves.svg)](https://pypi.org/project/tensorwaves)
[![Conda package](https://anaconda.org/conda-forge/tensorwaves/badges/version.svg)](https://anaconda.org/conda-forge/tensorwaves)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/tensorwaves)](https://pypi.org/project/tensorwaves)

[![Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ComPWA/tensorwaves/stable?filepath=docs/usage)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ComPWA/tensorwaves/blob/stable)
[![Open in Visual Studio Code](https://img.shields.io/badge/vscode-open-blue?logo=visualstudiocode)](https://open.vscode.dev/ComPWA/tensorwaves)
[![GitPod](https://img.shields.io/badge/gitpod-open-blue?logo=gitpod)](https://gitpod.io/#https://github.com/ComPWA/tensorwaves)

[![Documentation build status](https://readthedocs.org/projects/tensorwaves/badge/?version=latest)](https://tensorwaves.readthedocs.io)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ComPWA/tensorwaves/main.svg)](https://results.pre-commit.ci/latest/github/ComPWA/tensorwaves/main)
[![pytest](https://github.com/ComPWA/tensorwaves/workflows/pytest/badge.svg)](https://github.com/ComPWA/tensorwaves/actions?query=branch%3Amain+workflow%3Apytest)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy.readthedocs.io)
[![Test coverage](https://codecov.io/gh/ComPWA/tensorwaves/branch/main/graph/badge.svg)](https://codecov.io/gh/ComPWA/tensorwaves)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/db8f89e5588041d8a995968262c224ef)](https://www.codacy.com/gh/ComPWA/tensorwaves)
[![Spelling checked](https://img.shields.io/badge/cspell-checked-brightgreen.svg)](https://github.com/streetsidesoftware/cspell/tree/master/packages/cspell)
[![code style: prettier](https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square)](https://github.com/prettier/prettier)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort)

TensorWaves is a fitter package that optimizes mathematical models to data
samples. The models can be any _symbolic_ mathematical expression that is then
converted to any computational backend. In addition, TensorWaves provides
functionality to generate toy Monte Carlo data samples. The package is
developed in parallel with [AmpForm](https://github.com/ComPWA/ampform), which
implements physics models, but its 'lambdifying' mechanisms can in principle be
used independently.

All documentation can be found on
[tensorwaves.rtfd.io](https://tensorwaves.readthedocs.io).

For an overview of **upcoming releases and planned functionality**, see
[here](https://github.com/ComPWA/tensorwaves/milestones?direction=asc&sort=title&state=open).

## Contribute

See [`CONTRIBUTING.md`](./CONTRIBUTING.md)
