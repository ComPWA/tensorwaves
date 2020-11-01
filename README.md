# TensorWaves

[![Documentation build status](https://readthedocs.org/projects/tensorwaves/badge/?version=latest)](https://tensorwaves.readthedocs.io)
[![Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ComPWA/tensorwaves/stable?filepath=docs/usage)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ComPWA/expertsystem/blob/stable)
[![GPLv3+ license](https://img.shields.io/badge/License-GPLv3+-blue.svg)](https://www.gnu.org/licenses/gpl-3.0-standalone.html)
[![GitPod](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/ComPWA/expertsystem)
<br>
[![PyPI package](https://badge.fury.io/py/tensorwaves.svg)](https://pypi.org/project/tensorwaves)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/tensorwaves)](https://pypi.org/project/tensorwaves)
[![Checked with mypy](./docs/mypy_badge.svg)](http://mypy-lang.org) <br>
[![CI status](https://github.com/ComPWA/tensorwaves/workflows/CI/badge.svg)](https://github.com/ComPWA/tensorwaves/actions?query=branch%3Amaster+workflow%3ACI)
[![Test coverage](https://codecov.io/gh/ComPWA/tensorwaves/branch/master/graph/badge.svg)](https://codecov.io/gh/ComPWA/tensorwaves)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/db8f89e5588041d8a995968262c224ef)](https://www.codacy.com/gh/ComPWA/tensorwaves)
<br>
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)](https://github.com/pre-commit/pre-commit)
[![Prettier](https://camo.githubusercontent.com/687a8ae8d15f9409617d2cc5a30292a884f6813a/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f636f64655f7374796c652d70726574746965722d6666363962342e7376673f7374796c653d666c61742d737175617265)](https://prettier.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort)

TensorWaves is a Python package for doing Partial Wave Analysis with
[TensorFlow](https://www.tensorflow.org/) as computational backend. The package
originates from [pycompwa](https://compwa.github.io/), which did its
computations through ([ComPWA](https://github.com/ComPWA/ComPWA)) ― ComPWA's
function tree is similar to TensorFlow graphs and can therefore be replaced by
the tools that TensorFlow already offers. The decision to completely migrate
ComPWA to TensorFlow has been further inspired by
[TensorFlowAnalysis](https://gitlab.cern.ch/poluekt/TensorFlowAnalysis) and
[zfit](https://github.com/zfit/zfit/).

All documentation can be found on
[tensorwaves.rtfd.io](https://pwa.readthedocs.io/projects/tensorwaves/).

For an overview of **upcoming releases and planned functionality**, see
[here](https://github.com/ComPWA/tensorwaves/milestones?direction=asc&sort=title&state=open).

## Contribute

See [`CONTRIBUTING.md`](./CONTRIBUTING.md)
