"""Collection of fit optimizers.

Optimizers 'fit' a `.Model` to some data sample with regard to `.Estimator` as
loss function.
"""

__all__ = ["minuit", "scipy", "callbacks"]

from . import callbacks, minuit, scipy
