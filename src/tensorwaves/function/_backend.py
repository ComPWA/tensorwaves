"""Computational back-end handling."""
from __future__ import annotations

from functools import partial
from typing import Callable


def find_function(function_name: str, backend: str) -> Callable:
    backend_modules = get_backend_modules(backend)
    if isinstance(backend_modules, dict) and function_name in backend_modules:
        return backend_modules[function_name]
    if isinstance(backend_modules, (tuple, list)):
        for module in backend_modules:
            if isinstance(module, dict):
                module_dict = module
            else:
                module_dict = module.__dict__
            if function_name in module_dict:
                return module_dict[function_name]
    raise ValueError(
        f'Could not find function "{function_name}" in backend "{backend}"'
    )


def get_backend_modules(backend: str | tuple | dict) -> str | tuple | dict:
    """Preprocess the backend argument passed to `~sympy.utilities.lambdify.lambdify`.

    In `~sympy.utilities.lambdify.lambdify` the backend is specified via the
    :code:`modules` argument. Several back-ends can be specified by passing a
    `tuple` or dict`.
    """
    # pylint: disable=import-outside-toplevel
    if isinstance(backend, str):
        if backend == "jax":
            try:
                from jax import numpy as jnp
                from jax import scipy as jsp
                from jax.config import config
            except ImportError:  # pragma: no cover
                raise_missing_module_error("jax", extras_require="jax")

            config.update("jax_enable_x64", True)
            return (jnp, jsp.special)
        if backend in {"numpy", "numba"}:
            import numpy as np

            return np, np.__dict__
            # returning only np.__dict__ does not work well with conditionals
        if backend in {"tensorflow", "tf"}:
            try:
                # pylint: disable=import-error, no-name-in-module
                # pyright: reportMissingImports=false
                import tensorflow as tf
                import tensorflow.experimental.numpy as tnp
                from tensorflow.python.ops.numpy_ops import np_config
            except ImportError:  # pragma: no cover
                raise_missing_module_error("tensorflow", extras_require="tf")

            np_config.enable_numpy_behavior()

            return tnp.__dict__, tf

    return backend


def jit_compile(backend: str) -> Callable[[Callable], Callable]:
    # pylint: disable=import-outside-toplevel
    backend = backend.lower()
    if backend == "jax":
        try:
            import jax
        except ImportError:  # pragma: no cover
            raise_missing_module_error("jax", extras_require="jax")
        return jax.jit

    if backend == "numba":
        try:
            import numba  # pylint: disable=import-error
        except ImportError:  # pragma: no cover
            raise_missing_module_error("numba", extras_require="numba")
        return partial(numba.jit, forceobj=True, parallel=True)

    return lambda x: x


def raise_missing_module_error(
    module_name: str, *, extras_require: str = ""
) -> None:
    """Raise an `ImportError` with install instructions.

    >>> raise_missing_module_error("missing")
    Traceback (most recent call last):
        ...
    ImportError: Module missing not installed.
    >>> raise_missing_module_error("missing", extras_require="extras")
    Traceback (most recent call last):
        ...
    ImportError: Module missing not installed. Reinstall tensorwaves with:
    <BLANKLINE>
      pip install tensorwaves[extras]
    <BLANKLINE>
    """
    error_message = f"Module {module_name} not installed."
    if extras_require:
        error_message += (
            " Reinstall tensorwaves with:\n\n"
            f"  pip install tensorwaves[{extras_require}]\n"
        )
    raise ImportError(error_message)
