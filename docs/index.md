<!-- cspell:ignore Ampli -->

# Welcome to TensorWaves!

```{title} Welcome

```

[![10.5281/zenodo.5526650](https://zenodo.org/badge/doi/10.5281/zenodo.5526650.svg)](https://doi.org/10.5281/zenodo.5526650)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/tensorwaves)](https://pypi.org/project/tensorwaves)
{{ '[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ComPWA/tensorwaves/blob/{})'.format(branch) }}
{{ '[![Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ComPWA/tensorwaves/{}?furlpath=lab)'.format(branch) }}

:::{margin}

TensorWaves originates from [`pycompwa`](https://github.com/ComPWA/pycompwa), which did its computations through a function tree that is similar to [TensorFlow graphs](https://www.tensorflow.org/tensorboard/graphs) and [SymPy expression trees](https://docs.sympy.org/latest/tutorial/manipulation.html). The decision to use Python only has been further inspired by [TensorFlowAnalysis](https://gitlab.cern.ch/poluekt/TensorFlowAnalysis), [AmpliTF](https://github.com/apoluekt/AmpliTF) and [zfit](https://github.com/zfit/zfit).

:::

TensorWaves is an optimizer package that can 'fit' mathematical models (expressed with
{mod}`sympy`) to data distributions using a variety of computational back-ends,
optimizers, and estimator functions. In addition, the {mod}`tensorwaves.data` module
helps one to generate data distributions for those mathematical expressions.

The package is developed in parallel with {doc}`AmpForm <ampform:index>`, which
implements physics models for amplitude analysis, but its mechanisms for creating
computational backend functions can in principle be used independently.

<!-- prettier-ignore -->
::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item}

```{button-ref} usage
:ref-type: doc
:color: primary
:expand:
:outline:
:shadow:
```

:::

:::{grid-item}

```{button-ref} amplitude-analysis
:ref-type: doc
:color: primary
:expand:
:outline:
:shadow:
```

:::

::::

```{toctree}
---
maxdepth: 3
---
install
usage
amplitude-analysis
```

```{toctree}
---
hidden:
maxdepth: 2
---
API <api/tensorwaves>
Continuous benchmarks <https://compwa.github.io/tensorwaves>
Changelog <https://github.com/ComPWA/tensorwaves/releases>
Upcoming features <https://github.com/ComPWA/tensorwaves/milestones?direction=asc&sort=title&state=open>
Help developing <https://compwa.github.io/develop>
```

```{toctree}
---
caption: Related projects
hidden:
---
QRules <https://qrules.readthedocs.io>
AmpForm <https://ampform.readthedocs.io>
PWA Pages <https://pwa.readthedocs.io>
ComPWA project <https://compwa.github.io>
```
