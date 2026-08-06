"""Configure optional computational backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from types import ModuleType

Precision = Literal["float32", "float64"]


def configure(
    *,
    jax_precision: Literal["float32", "float64"] | None = None,
    tensorflow_precision: Literal["float32", "float64"] | None = None,
) -> None:
    """Set the precision used by computational backends.

    Call this function before creating backend arrays or TensorWaves functions. By
    default, TensorWaves uses 64-bit precision for JAX and TensorFlow. TensorWaves
    respects ``JAX_ENABLE_X64`` if ``jax_precision`` is not specified.
    """
    if jax_precision is not None:
        _validate_precision("jax_precision", jax_precision)
        _jax_config.precision = jax_precision
        if _jax_config.initialized:
            jax = import_module("jax")
            _set_jax_precision(jax, jax_precision)
    if tensorflow_precision is not None:
        _validate_precision("tensorflow_precision", tensorflow_precision)
        _tensorflow_config.precision = tensorflow_precision
        if _tensorflow_config.initialized:
            tf = import_module("tensorflow")
            _set_tensorflow_precision(tf, tensorflow_precision)


def _initialize_jax() -> ModuleType:
    jax = import_module("jax")

    if not _jax_config.initialized:
        if _jax_config.precision is not None:
            _set_jax_precision(jax, _jax_config.precision)
        elif "JAX_ENABLE_X64" not in os.environ:
            _set_jax_precision(jax, "float64")
        _jax_config.initialized = True
    return jax


def _set_jax_precision(jax: ModuleType, precision: Precision) -> None:
    jax.config.update("jax_enable_x64", precision == "float64")


def _initialize_tensorflow() -> ModuleType:
    tf = import_module("tensorflow")

    if not _tensorflow_config.initialized:
        precision = _tensorflow_config.precision or "float64"
        _set_tensorflow_precision(tf, precision)
        _tensorflow_config.initialized = True
    return tf


def _set_tensorflow_precision(tensorflow: ModuleType, precision: Precision) -> None:
    tensorflow.experimental.numpy.experimental_enable_numpy_behavior(
        prefer_float32=precision == "float32"
    )


def _tensorflow_float_dtype(tensorflow: ModuleType) -> object:
    if _tensorflow_config.precision == "float32":
        return tensorflow.float32
    return tensorflow.float64


def _validate_precision(name: str, precision: object) -> None:
    if precision not in {"float32", "float64"}:
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
