# pylint: disable=import-error, import-outside-toplevel
# pyright: reportMissingImports=false
from tensorwaves.function._backend import find_function


def test_find_function_jax():
    import jax.numpy as jnp

    assert find_function("array", backend="jax") is jnp.array
    assert find_function("linspace", backend="jax") is jnp.linspace
    assert find_function("mean", backend="jax") is jnp.mean


def test_find_function_numpy():
    import numpy as np

    assert find_function("array", backend="numpy") is np.array
    assert find_function("linspace", backend="numpy") is np.linspace
    assert find_function("log", backend="numpy") is np.log
    assert find_function("mean", backend="numpy") is np.mean
    assert find_function("mean", backend="numba") is np.mean


def test_find_function_tf():
    import tensorflow as tf
    import tensorflow.experimental.numpy as tnp

    assert find_function("array", backend="tf") is tnp.array
    assert find_function("linspace", backend="tf") is tnp.linspace
    assert find_function("mean", backend="tf") is tnp.mean
    assert find_function("Tensor", backend="tf") is tf.Tensor
