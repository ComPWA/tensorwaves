from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sympy as sp

from tensorwaves import configure
from tensorwaves.function.sympy import create_function

if TYPE_CHECKING:
    from typing import Literal

    from tensorwaves.config import Precision
    from tensorwaves.interface import DataSample

    Backend = Literal["jax", "tensorflow"]


@pytest.mark.benchmark(group="precision")
@pytest.mark.parametrize("backend", ["jax", "tensorflow"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_precision(benchmark, backend: Backend, precision: Precision) -> None:
    _configure_backend(backend, precision)
    x = sp.Symbol("x")
    function = create_function(sp.sin(x) ** 2 + sp.exp(-x), backend=backend)
    data = _create_data(backend)
    result = benchmark(lambda: _evaluate(function, data, backend))
    _assert_precision(backend, precision, data, result)


def _configure_backend(backend: Backend, precision: Precision) -> None:
    if backend == "jax":
        configure(jax_precision=precision)
    else:
        configure(tensorflow_precision=precision)


def _create_data(backend: Backend):
    if backend == "jax":
        import jax.numpy as jnp

        return {"x": jnp.linspace(0, 10, num=1_000_000).block_until_ready()}

    import tensorflow.experimental.numpy as tnp  # ty: ignore[unresolved-import]

    return {"x": tnp.linspace(0, 10, num=1_000_000)}


def _evaluate(function, data: DataSample, backend: Backend):
    result = function(data)
    if backend == "jax":
        result.block_until_ready()
    return result


def _assert_precision(
    backend: Backend, precision: Precision, data: dict, result
) -> None:
    assert data["x"].dtype.name == precision
    assert result.dtype.name == precision
    if backend == "jax":
        import jax

        assert jax.config.x64_enabled == (precision == "float64")
    else:
        import tensorflow.experimental.numpy as tnp  # ty: ignore[unresolved-import]

        assert tnp.asarray(1.0).dtype.name == precision
