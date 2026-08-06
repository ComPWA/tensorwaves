"""Configure optional computational backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


@dataclass
class _JaxConfig:
    enable_x64: bool | None = None
    initialized: bool = False


_jax_config = _JaxConfig()


def configure(*, jax_enable_x64: bool) -> None:
    """Set the precision used by the JAX backend.

    Call this function before creating JAX arrays or TensorWaves functions. If it is
    not called, TensorWaves respects ``JAX_ENABLE_X64`` and otherwise enables 64-bit
    precision.
    """
    if not isinstance(jax_enable_x64, bool):
        msg = "jax_enable_x64 must be a bool"
        raise TypeError(msg)
    _jax_config.enable_x64 = jax_enable_x64
    if _jax_config.initialized:
        import jax  # ruff: ignore[import-outside-top-level]

        _set_jax_precision(jax, jax_enable_x64)


def _initialize_jax() -> ModuleType:
    import jax  # ruff: ignore[import-outside-top-level]

    if not _jax_config.initialized:
        if _jax_config.enable_x64 is not None:
            _set_jax_precision(jax, _jax_config.enable_x64)
        elif "JAX_ENABLE_X64" not in os.environ:
            _set_jax_precision(jax, True)
        _jax_config.initialized = True
    return jax


def _set_jax_precision(jax: ModuleType, enable_x64: bool) -> None:
    jax.config.update("jax_enable_x64", enable_x64)
