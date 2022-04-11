<!-- cspell:ignore Ampli -->

# Welcome to TensorWaves!

```{title} Welcome

```

<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
[![10.5281/zenodo.5526650](https://zenodo.org/badge/doi/10.5281/zenodo.5526650.svg)](https://doi.org/10.5281/zenodo.5526650)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/tensorwaves)](https://pypi.org/project/tensorwaves)
{{ '[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ComPWA/tensorwaves/blob/{})'.format(branch) }}
{{ '[![Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ComPWA/tensorwaves/{}?filepath=docs/usage)'.format(branch) }}
<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->

:::{margin}

TensorWaves originates from [`pycompwa`](https://compwa.github.io), which did
its computations through a function tree that is similar to
[TensorFlow graphs](https://www.tensorflow.org/tensorboard/graphs) and
[SymPy expression trees](https://docs.sympy.org/latest/tutorial/manipulation.html).
The decision to use Python only has been further inspired by
[TensorFlowAnalysis](https://gitlab.cern.ch/poluekt/TensorFlowAnalysis),
[AmpliTF](https://github.com/apoluekt/AmpliTF) and
[zfit](https://github.com/zfit/zfit).

:::

TensorWaves is an optimizer package that can 'fit' mathematical models
(expressed with {mod}`sympy`) to data distributions using a variety of
computational back-ends, optimizers, and estimator functions. In addition, the
{mod}`tensorwaves.data` module helps one to generate data distributions for
those mathematical expressions.

The package is developed in parallel with {doc}`AmpForm <ampform:index>`, which
implements physics models for amplitude analysis, but its mechanisms for
creating computational backend functions can in principle be used
independently.

:::{panels}

```{link-button} usage
:type: ref
:text: General examples
:classes: btn-outline-primary btn-block
```

---

```{link-button} amplitude-analysis
:type: ref
:text: Amplitude analysis
:classes: btn-outline-primary btn-block
```

:::

```{rubric} Table of contents

```

```{toctree}
---
maxdepth: 3
---
install
usage
amplitude-analysis
API <api/tensorwaves>
Continuous benchmarks <https://compwa.github.io/tensorwaves>
Changelog <https://github.com/ComPWA/tensorwaves/releases>
Upcoming features <https://github.com/ComPWA/tensorwaves/milestones?direction=asc&sort=title&state=open>
Help developing <https://compwa-org.rtfd.io/en/stable/develop.html>
```

```{toctree}
---
caption: Related projects
hidden:
---
AmpForm <https://ampform.readthedocs.io>
QRules <https://qrules.readthedocs.io>
PWA Pages <https://pwa.readthedocs.io>
```

```{toctree}
---
caption: ComPWA Organization
hidden:
---
Website <https://compwa-org.readthedocs.io>
GitHub Repositories <https://github.com/ComPWA>
About <https://compwa-org.readthedocs.io/en/stable/about.html>
```
