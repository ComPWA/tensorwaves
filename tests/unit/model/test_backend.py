from tensorwaves.model._backend import find_function


def test_find_function():
    # pylint: disable=import-error, import-outside-toplevel
    # pyright: reportMissingImports=false
    import jax.numpy as jnp
    import numpy as np
    import tensorflow as tf
    import tensorflow.experimental.numpy as tnp

    assert find_function("numpy", "array") is np.array
    assert find_function("numpy", "linspace") is np.linspace
    assert find_function("numpy", "log") is np.log
    assert find_function("numpy", "mean") is np.mean
    assert find_function("numba", "mean") is np.mean

    assert find_function("jax", "array") is jnp.array
    assert find_function("jax", "linspace") is jnp.linspace
    assert find_function("jax", "mean") is jnp.mean

    assert find_function("tf", "array") is tnp.array
    assert find_function("tf", "linspace") is tnp.linspace
    assert find_function("tf", "mean") is tnp.mean
    assert find_function("tf", "Tensor") is tf.Tensor
