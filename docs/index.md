# Welcome to TensorWaves!

```{title} Welcome

```

[![PyPI package](https://badge.fury.io/py/tensorwaves.svg)](https://pypi.org/project/tensorwaves)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/tensorwaves)](https://pypi.org/project/tensorwaves)
[![Test coverage](https://codecov.io/gh/ComPWA/tensorwaves/branch/master/graph/badge.svg)](https://codecov.io/gh/ComPWA/tensorwaves)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/db8f89e5588041d8a995968262c224ef)](https://www.codacy.com/gh/ComPWA/tensorwaves)

````{margin}
```{tip}
For an overview of upcoming releases and planned functionality, see
[here](https://github.com/ComPWA/tensorwaves/milestones?direction=asc&sort=title&state=open).
```
````

For a quick demo of the TensorWaves workflow, see the
[workflow notebooks on binder](https://mybinder.org/v2/gh/ComPWA/tensorwaves/master?filepath=examples%2Fworkflow).

TensorWaves is a Python package for doing Partial Wave Analysis with
[TensorFlow](https://www.tensorflow.org) as computational backend. The package
originates from [pycompwa](pycompwa), which did its computations through
[ComPWA](https://github.com/ComPWA/ComPWA) — ComPWA's function tree is similar
to TensorFlow graphs and can therefore be replaced by the tools that TensorFlow
already offers. The decision to completely migrate ComPWA to TensorFlow has
been further inspired by
[TensorFlowAnalysis](https://gitlab.cern.ch/poluekt/TensorFlowAnalysis)/[AmpliTF](https://github.com/apoluekt/AmpliTF)
and [zfit](https://github.com/zfit/zfit)

```{toctree}
---
maxdepth: 2
---
install
usage
API <api/tensorwaves>
Develop <https://pwa.readthedocs.io/develop.html>
```

- {ref}`Python API <modindex>`
- {ref}`General Index <genindex>`
- {ref}`Search <search>`

```{toctree}
---
caption: Related projects
hidden:
---
Expert System <http://expertsystem.readthedocs.io>
PWA Pages <http://pwa.readthedocs.io>
```
