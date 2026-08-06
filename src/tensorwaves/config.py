"""Configure optional computational backends."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

Precision = Literal["float32", "float64"]


def configure(
    *,
    jax_precision: Precision | None = None,
    tensorflow_precision: Precision | None = None,
) -> None:
    """Set the precision used by computational backends.

    Call this function before creating backend arrays or TensorWaves functions. By
    default, TensorWaves uses 64-bit precision for JAX and TensorFlow. TensorWaves
    respects ``JAX_ENABLE_X64`` if ``jax_precision`` is not specified.
    """
    _validate_precision("jax_precision", jax_precision)
    _validate_precision("tensorflow_precision", tensorflow_precision)
    if jax_precision is not None:
        _jax_config.precision = jax_precision
        _configure_imported_module("jax", _set_jax_precision, jax_precision)
    if tensorflow_precision is not None:
        _tensorflow_config.precision = tensorflow_precision
        _configure_imported_module(
            "tensorflow", _set_tensorflow_precision, tensorflow_precision
        )


def _configure_imported_module(
    module_name: str,
    set_precision: Callable[[ModuleType, Precision], None],
    precision: Precision,
) -> None:
    """Set the precision on a backend that has already been imported.

    A backend that has not been imported yet is configured on first use, so that
    :func:`configure` never triggers a backend import itself.
    """
    module = sys.modules.get(module_name)
    if module is not None:
        set_precision(module, precision)


def _initialize_jax() -> ModuleType:
    jax = import_module("jax")

    if not _jax_config.initialized:
        precision = _jax_config.precision
        if precision is None:
            precision = _precision_from_flag(os.environ.get("JAX_ENABLE_X64", "1"))
        _set_jax_precision(jax, precision)
        _jax_config.initialized = True
    return jax


def _set_jax_precision(jax: ModuleType, precision: Precision) -> None:
    jax.config.update("jax_enable_x64", precision == "float64")


def _precision_from_flag(value: str) -> Precision:
    """Interpret a JAX-style boolean environment variable value.

    >>> _precision_from_flag("1"), _precision_from_flag("false")
    ('float64', 'float32')
    """
    return "float64" if value.strip().lower() in {"1", "true", "yes"} else "float32"


def _initialize_tensorflow() -> ModuleType:
    tf = import_module("tensorflow")

    if not _tensorflow_config.initialized:
        _set_tensorflow_precision(tf, _tensorflow_precision())
        _tensorflow_config.initialized = True
    return tf


def _set_tensorflow_precision(tensorflow: ModuleType, precision: Precision) -> None:
    tensorflow.experimental.numpy.experimental_enable_numpy_behavior(
        prefer_float32=precision == "float32"
    )


def _tensorflow_precision() -> Precision:
    return _tensorflow_config.precision or "float64"


def _validate_precision(name: str, precision: object) -> None:
    if precision is not None and precision not in get_args(Precision):
        msg = f"{name} must be 'float32', 'float64', or None"
        raise ValueError(msg)


@dataclass
class _JaxConfig:
    precision: Precision | None = None
    initialized: bool = False


@dataclass
class _TensorFlowConfig:
    precision: Precision | None = None
    initialized: bool = False


_jax_config = _JaxConfig()
_tensorflow_config = _TensorFlowConfig()
