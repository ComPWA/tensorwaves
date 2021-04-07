"""A model optimization package for Partial Wave Analysis.

The `tensorwaves` package contains four main ingredients:

  Model lambdifying (`tensorwaves.model`)
    ― Convert arbitrary mathematical expressions to an computational backends
    like `numpy`, `jax <jax.jit>`, and `tensorflow <tf.Tensor>`. This module
    works best with `sympy` expressions as input.

  Data generation (`tensorwaves.data`)
    ― Generate phase space samples and toy Monte Carlo samples based on an
    intensity `.Model`. This module works best in combination with `ampform`.

  Model optimization (`tensorwaves.optimizer` and `~.estimator`)
    ― Optimize the `.Model` with respect to some data sample and an
    `.Estimator` (loss function).

The `.interfaces` module defines how the main classes interact.
"""

__all__ = [
    "data",
    "estimator",
    "model",
    "optimizer",
]

from . import data, estimator, model, optimizer
