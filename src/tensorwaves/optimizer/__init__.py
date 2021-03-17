"""Collection of fit optimizers.

Optimizers 'fit' a `.Model` to some data sample with regard to `.Estimator` as
loss function.
"""

__all__ = [
    "callbacks",
    "minuit",
]

from . import callbacks, minuit

try:
    from . import scipy  # noqa: F401

    __all__.append("scipy")
except ImportError:
    pass
