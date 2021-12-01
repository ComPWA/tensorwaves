from tensorwaves._backend import find_function


def test_find_function():
    # pylint: disable=import-error, import-outside-toplevel
    # pyright: reportMissingImports=false
    import jax.numpy as jnp
    import numpy as np
    import tensorflow.experimental.numpy as tnp

    assert find_function("numpy", "mean") is np.mean
    assert find_function("numpy", "log") is np.log
    assert find_function("tf", "mean") is tnp.mean
    assert find_function("jax", "mean") is jnp.mean
