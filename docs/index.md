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

TensorWaves is a fitter package that optimizes mathematical models to data
samples. The models can be any mathematical expression (best expressed with
{mod}`sympy`) that is then converted to any computational backend. In addition,
the {mod}`tensorwaves.data` module allows one to generate toy data samples.

```{link-button} usage
---
classes: btn-outline-primary btn-block
type: ref
text: Click here for a quick demo
---
```

The package originates from {mod}`pycompwa`, which did its computations through
[ComPWA](https://github.com/ComPWA/ComPWA). ComPWA's function tree is similar
to for instance
[TensorFlow graphs](https://www.tensorflow.org/tensorboard/graphs) and
[SymPy expression trees](https://docs.sympy.org/latest/tutorial/manipulation.html).
The decision to use Python only has been further inspired by
[TensorFlowAnalysis](https://gitlab.cern.ch/poluekt/TensorFlowAnalysis)/[AmpliTF](https://github.com/apoluekt/AmpliTF)
and [zfit](https://github.com/zfit/zfit).

## Table of Contents

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
