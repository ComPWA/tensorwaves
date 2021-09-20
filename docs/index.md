<!-- cspell:ignore Ampli -->

# Welcome to TensorWaves!

```{title} Welcome

```

[![PyPI package](https://badge.fury.io/py/tensorwaves.svg)](https://pypi.org/project/tensorwaves)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/tensorwaves)](https://pypi.org/project/tensorwaves)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ComPWA/tensorwaves/blob/stable)
[![Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ComPWA/tensorwaves/stable?filepath=docs/usage)

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

TensorWaves is a fitter package that optimizes mathematical models to data
samples. The models can be any mathematical expression (best expressed with
{mod}`sympy`) that is then converted to any computational backend. In addition,
the {mod}`tensorwaves.data` module allows one to generate toy data samples. The
package is developed in parallel with {doc}`AmpForm <ampform:index>`, which
implements physics models, but its 'lambdifying' mechanisms can in principle be
used independently.

```{link-button} usage
---
classes: btn-outline-primary btn-block
type: ref
text: Click here for a quick demo
---
```

```{rubric} Table of contents

```

```{toctree}
---
maxdepth: 3
---
install
usage
API <api/tensorwaves>
Changelog <https://github.com/ComPWA/tensorwaves/releases>
Upcoming features <https://github.com/ComPWA/tensorwaves/milestones?direction=asc&sort=title&state=open>
Help developing <https://compwa-org.rtfd.io/en/stable/develop.html>
```

- {ref}`Python API <modindex>`
- {ref}`General Index <genindex>`
- {ref}`Search <search>`

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
