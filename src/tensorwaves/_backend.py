"""Computational back-end handling."""


from functools import partial
from typing import Callable, Union


def find_function(
    backend: Union[str, tuple, dict], function_name: str
) -> Callable:
    backend_modules = get_backend_modules(backend)
    if isinstance(backend_modules, dict) and function_name in backend_modules:
        return backend_modules[function_name]
    if isinstance(backend_modules, (tuple, list)):
        for module in backend_modules:
            if function_name in module.__dict__:
                return module.__dict__[function_name]
    raise ValueError(f"Could not find function {function_name} in backend")


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
            # pyright: reportMissingImports=false
            import tensorflow.experimental.numpy as tnp

            return tnp.__dict__

    return backend


def jit_compile(backend: str) -> Callable:
    # pylint: disable=import-outside-toplevel
    backend = backend.lower()
    if backend == "jax":
        import jax

        return jax.jit

    if backend == "numba":
        import numba

        return partial(numba.jit, forceobj=True, parallel=True)

    return lambda x: x
