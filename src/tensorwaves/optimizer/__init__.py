"""Collection of fit optimizers.

Optimizers 'fit' a `.ParametrizedFunction` to some data sample with regard to
`.Estimator` as loss function.
"""

__all__ = [
    "callbacks",
    "minuit",
]

# pyright: reportUnusedImport=false
from . import callbacks, minuit
from .minuit import Minuit2  # noqa: F401

try:
    from . import scipy
    from .scipy import ScipyMinimizer  # noqa: F401

    __all__ += [
        "scipy",
    ]
except ImportError:  # pragma: no cover
    pass
