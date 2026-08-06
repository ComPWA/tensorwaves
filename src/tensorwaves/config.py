"""Configure optional computational backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


@dataclass
class _JaxConfig:
    enable_x64: bool | None = None
    initialized: bool = False


_jax_config = _JaxConfig()


@dataclass
class _TensorFlowConfig:
    prefer_float32: bool | None = None
    initialized: bool = False


_tensorflow_config = _TensorFlowConfig()


def configure(
    *,
    jax_enable_x64: bool | None = None,
    tensorflow_prefer_float32: bool | None = None,
) -> None:
    """Set the precision used by computational backends.

    Call this function before creating backend arrays or TensorWaves functions. By
    default, TensorWaves uses 64-bit precision for JAX and TensorFlow. TensorWaves
    respects ``JAX_ENABLE_X64`` if ``jax_enable_x64`` is not specified.
    """
    if jax_enable_x64 is not None:
        if not isinstance(jax_enable_x64, bool):
            msg = "jax_enable_x64 must be a bool or None"
            raise TypeError(msg)
        _jax_config.enable_x64 = jax_enable_x64
        if _jax_config.initialized:
            jax = import_module("jax")
            _set_jax_precision(jax, jax_enable_x64)
    if tensorflow_prefer_float32 is not None:
        if not isinstance(tensorflow_prefer_float32, bool):
            msg = "tensorflow_prefer_float32 must be a bool or None"
            raise TypeError(msg)
        _tensorflow_config.prefer_float32 = tensorflow_prefer_float32
        if _tensorflow_config.initialized:
            tf = import_module("tensorflow")
            _enable_tensorflow_numpy_behavior(tf, tensorflow_prefer_float32)


def _initialize_jax() -> ModuleType:
    jax = import_module("jax")

    if not _jax_config.initialized:
        if _jax_config.enable_x64 is not None:
            _set_jax_precision(jax, _jax_config.enable_x64)
        elif "JAX_ENABLE_X64" not in os.environ:
            _set_jax_precision(jax, True)
        _jax_config.initialized = True
    return jax


def _set_jax_precision(jax: ModuleType, enable_x64: bool) -> None:
    jax.config.update("jax_enable_x64", enable_x64)


def _initialize_tensorflow() -> ModuleType:
    tf = import_module("tensorflow")

    if not _tensorflow_config.initialized:
        prefer_float32 = _tensorflow_config.prefer_float32 is True
        _enable_tensorflow_numpy_behavior(tf, prefer_float32)
        _tensorflow_config.initialized = True
    return tf


def _enable_tensorflow_numpy_behavior(
    tensorflow: ModuleType, prefer_float32: bool
) -> None:
    tensorflow.experimental.numpy.experimental_enable_numpy_behavior(
        prefer_float32=prefer_float32
    )


def _tensorflow_float_dtype(tensorflow: ModuleType) -> object:
    if _tensorflow_config.prefer_float32 is True:
        return tensorflow.float32
    return tensorflow.float64
