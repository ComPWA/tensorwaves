"""Computational back-end handling."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING
from warnings import warn

from tensorwaves.config import _initialize_jax, _initialize_tensorflow

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import NoReturn, ParamSpec, TypeVar

    P = ParamSpec("P")
    T = TypeVar("T")


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
    msg = f'Could not find function "{function_name}" in backend "{backend}"'
    raise ValueError(msg)


def get_backend_modules(backend: str | tuple | dict) -> str | tuple | dict:
    """Preprocess the backend argument passed to `~sympy.utilities.lambdify.lambdify`.

    In `~sympy.utilities.lambdify.lambdify` the backend is specified via the
    :code:`modules` argument. Several back-ends can be specified by passing a `tuple` or
    dict`.
    """
    if isinstance(backend, str):
        if backend == "jax":
            try:
                _initialize_jax()
                import jax.numpy as jnp
                import jax.scipy as jsp
            except ImportError:  # pragma: no cover
                raise_missing_module_error("jax", extras_require="jax")
            return jnp, jsp.special
        if backend in {"numpy", "numba"}:
            import numpy as np

            return np, np.__dict__
            # returning only np.__dict__ does not work well with conditionals
        if backend in {"tensorflow", "tf"}:
            try:
                tf = _initialize_tensorflow()
                tnp = tf.experimental.numpy
            except ImportError:  # pragma: no cover
                raise_missing_module_error("tensorflow", extras_require="tf")
            return tnp.__dict__, tf

    return backend


def get_jit_compile_dectorator(
    backend: str, use_jit: bool | None
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    if use_jit is None:
        backends_supporting_jit = {"jax", "numba"}
        if backend.lower() in backends_supporting_jit:
            return jit_compile(backend)
        return _do_not_compile
    if use_jit:
        return jit_compile(backend)
    return _do_not_compile


def jit_compile(backend: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    backend = backend.lower()
    if backend == "jax":
        try:
            import jax
        except ImportError:  # pragma: no cover
            raise_missing_module_error("jax", extras_require="jax")
        return jax.jit

    if backend == "numba":
        try:
            import numba
        except ImportError:  # pragma: no cover
            raise_missing_module_error("numba", extras_require="numba")
        return partial(numba.jit, forceobj=True, parallel=True)

    msg = f"Backend {backend} does not yet support JIT compilation"
    warn(msg, category=UserWarning, stacklevel=3)
    return _do_not_compile


def _do_not_compile(function: Callable[P, T]) -> Callable[P, T]:
    return function


def raise_missing_module_error(
    module_name: str, *, extras_require: str = ""
) -> NoReturn:
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
