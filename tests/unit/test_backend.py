from tensorwaves._backend import find_function


def test_find_function():
    # pylint: disable=import-error, import-outside-toplevel
    # pyright: reportMissingImports=false
    import jax.numpy as jnp
    import numpy as np
    import tensorflow.experimental.numpy as tnp

    assert find_function("mean", backend="numpy") is np.mean
    assert find_function("log", backend="numpy") is np.log
    assert find_function("mean", backend="tf") is tnp.mean
    assert find_function("mean", backend="jax") is jnp.mean
