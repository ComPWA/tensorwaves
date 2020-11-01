# Welcome to TensorWaves!

```{title} Welcome

```

<!-- prettier:disable -->

```{eval-rst}
.. list-table::
  :widths: auto
  :align: left

  * - .. image:: https://badge.fury.io/py/tensorwaves.svg
        :alt: PyPI package
        :target: https://pypi.org/project/tensorwaves
    - .. image:: https://img.shields.io/pypi/pyversions/tensorwaves
        :alt: Supported Python versions
        :target: https://pypi.org/project/tensorwaves
    - .. image:: https://codecov.io/gh/ComPWA/tensorwaves/branch/master/graph/badge.svg
        :alt: Test Coverage
        :target: https://codecov.io/gh/ComPWA/tensorwaves
    - .. image:: https://api.codacy.com/project/badge/Grade/db8f89e5588041d8a995968262c224ef
        :alt: Codacy Badge
        :target: https://www.codacy.com/gh/ComPWA/tensorwaves
```

<!-- prettier:enable -->

````{margin}
```{tip}
For an overview of upcoming releases and planned functionality, see
[here](https://github.com/ComPWA/expertsystem/milestones?direction=asc&sort=title&state=open)
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
develop
```

- {ref}`Python API <modindex>`
- {ref}`General Index <genindex>`
- {ref}`Search <search>`

```{toctree}
---
hidden:
---
api
```

```{toctree}
---
caption: Related projects
hidden:
---
Expert System <http://expertsystem.readthedocs.io>
PWA Pages <http://pwa.readthedocs.io>
```
