"""Collection of fit optimizers.

Optimizers 'fit' a `.ParametrizedFunction` to some data sample with regard to
`.Estimator` as loss function.
"""

__all__ = [
    "callbacks",
    "minuit",
]

from . import callbacks, minuit
from .minuit import Minuit2  # ruff:ignore[unused-import]

try:
    from . import scipy
    from .scipy import ScipyMinimizer  # ruff:ignore[unused-import]

    __all__ += [
        "scipy",
    ]
except ImportError:  # pragma: no cover
    pass
