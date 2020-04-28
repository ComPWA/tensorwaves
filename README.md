[![PyPI](https://badge.fury.io/py/tensorwaves.svg)](https://pypi.org/project/tensorwaves)
[![Travis CI](https://travis-ci.com/ComPWA/tensorwaves.svg?branch=master)](https://travis-ci.com/ComPWA/tensorwaves)
[![Test coverage](https://codecov.io/gh/ComPWA/tensorwaves/branch/master/graph/badge.svg)](https://codecov.io/gh/ComPWA/tensorwaves)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/db8f89e5588041d8a995968262c224ef)](https://www.codacy.com/gh/ComPWA/tensorwaves)
[![Documentation build status](https://readthedocs.org/projects/tensorwaves/badge/?version=latest)](https://pwa.readthedocs.io/projects/tensorwaves/en/latest/?badge=latest)

# TensorWaves

*This package is Work-In-Progress and currently unstable.*

TensorWaves is a Python package for doing Partial Wave Analysis with
[TensorFlow](https://www.tensorflow.org/) as computational backend. The package
originates from [pycompwa](https://compwa.github.io/) (see in particular the
[expert system](./tensorwaves/expertsystem)), which did its computations
through ([ComPWA](https://github.com/ComPWA/ComPWA)) ― ComPWA's function tree
is similar to TensorFlow graphs and can therefore be replaced by the tools that
TensorFlow already offers. The decision to completely migrate ComPWA to
TensorFlow has been further inspired by
[TensorFlowAnalysis](https://gitlab.cern.ch/poluekt/TensorFlowAnalysis) and
[zfit](https://github.com/zfit/zfit/).

All documentation can be found on
[pwa.readthedocs.io/projects/tensorwaves](https://pwa.readthedocs.io/projects/tensorwaves/)


## Installation

See
[tensorwaves.readthedocs.io/installation](https://pwa.readthedocs.io/projects/tensorwaves/installation)!

## Contribute

See [`CONTRIBUTING.md`](./CONTRIBUTING.md)
