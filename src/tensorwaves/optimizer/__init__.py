"""Collection of fit optimizers.

Optimizers 'fit' a `.ParametrizedFunction` to some data sample with regard to `.Estimator`
as loss function.
"""

__all__ = [
    "callbacks",
    "minuit",
]

# pyright: reportUnusedImport=false
from . import callbacks, minuit
from .minuit import Minuit2

try:
    from . import scipy
    from .scipy import ScipyMinimizer

    __all__ += [
        "scipy",
    ]
except ImportError:  # pragma: no cover
    pass
