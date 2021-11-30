"""A model optimization package for Partial Wave Analysis.

The `tensorwaves` package contains four main ingredients:

  Function creation (`tensorwaves.function`)
    ― Express arbitrary mathematical expressions as functions in different
    kinds of computational backends, like `numpy`, `jax <jax.jit>`, and
    `tensorflow <tf.Tensor>`. This module works best with `sympy` expressions
    as input.

  Data generation (`tensorwaves.data`)
    ― Generate phase space samples as well as hit-and-miss Monte Carlo samples
    for the input mathematical expression.

  Optimization (`tensorwaves.optimizer` and `~.estimator`)
    ― Optimize the `.Function` with respect to some data sample and an
    `.Estimator` (loss function).

The `.interface` module defines how the main classes interact.
"""

__all__ = [
    "data",
    "estimator",
    "function",
    "optimizer",
]

from . import data, estimator, function, optimizer
