# pyright: reportMissingImports=false
"""Computational back-end handling."""

from typing import Union


def get_backend_modules(
    backend: Union[str, tuple, dict],
) -> Union[str, tuple, dict]:
    """Preprocess the backend argument passed to `~sympy.utilities.lambdify.lambdify`.

    Note in `~sympy.utilities.lambdify.lambdify` the backend is specified via
    the :code:`modules` argument.
    """
    # pylint: disable=import-outside-toplevel
    if isinstance(backend, str):
        if backend == "jax":
            from jax import numpy as jnp
            from jax import scipy as jsp
            from jax.config import config

            config.update("jax_enable_x64", True)

            return (jnp, jsp.special)
        if backend in {"numpy", "numba"}:
            import numpy as np

            return (np, np.__dict__)
            # returning only np.__dict__ does not work well with conditionals
        if backend in {"tensorflow", "tf"}:
            # pylint: disable=import-error
            import tensorflow.experimental.numpy as tnp

            return tnp.__dict__

    return backend
