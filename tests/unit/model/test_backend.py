# pylint: disable=import-error
# pyright: reportMissingImports=false
import jax.numpy as jnp
import numpy as np
import tensorflow.experimental.numpy as tnp

from tensorwaves.model.backend import find_function


def test_find_function():
    assert find_function("numpy", "mean") is np.mean
    assert find_function("numpy", "log") is np.log
    assert find_function("tf", "mean") is tnp.mean
    assert find_function("jax", "mean") is jnp.mean
