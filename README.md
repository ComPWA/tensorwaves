[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ComPWA/tensorwaves/master?filepath=examples%2Fworkflow)
[![PyPI](https://badge.fury.io/py/tensorwaves.svg)](https://pypi.org/project/tensorwaves)
[![Travis CI](https://travis-ci.com/ComPWA/tensorwaves.svg?branch=master)](https://travis-ci.com/ComPWA/tensorwaves)
[![Test coverage](https://codecov.io/gh/ComPWA/tensorwaves/branch/master/graph/badge.svg)](https://codecov.io/gh/ComPWA/tensorwaves)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/db8f89e5588041d8a995968262c224ef)](https://www.codacy.com/gh/ComPWA/tensorwaves)
[![Documentation build status](https://readthedocs.org/projects/tensorwaves/badge/?version=latest)](https://pwa.readthedocs.io/projects/tensorwaves/en/latest/?badge=latest)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Prettier](https://camo.githubusercontent.com/687a8ae8d15f9409617d2cc5a30292a884f6813a/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f636f64655f7374796c652d70726574746965722d6666363962342e7376673f7374796c653d666c61742d737175617265)](https://prettier.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# TensorWaves

_For a quick demo of the TensorWaves workflow, see the
[workflow notebooks on binder](https://mybinder.org/v2/gh/ComPWA/tensorwaves/master?filepath=examples%2Fworkflow)._

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
[pwa.readthedocs.io/projects/tensorwaves](https://pwa.readthedocs.io/projects/tensorwaves/)

## Contribute

See [`CONTRIBUTING.md`](./CONTRIBUTING.md)
