from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import sympy as sp

from tensorwaves import configure
from tensorwaves.function.sympy import create_function

if TYPE_CHECKING:
    from tensorwaves.config import Precision


@pytest.mark.benchmark(group="precision")
@pytest.mark.parametrize("backend", ["jax", "tensorflow"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_precision(benchmark, backend: str, precision: Precision) -> None:
    _configure_backend(backend, precision)
    x = sp.Symbol("x")
    function = create_function(sp.sin(x) ** 2 + sp.exp(-x), backend=backend)
    data = {"x": np.linspace(0, 10, num=1_000_000, dtype=precision)}
    result = benchmark(lambda: _evaluate(function, data, backend))
    assert result.dtype.name == precision


def _configure_backend(backend: str, precision: Precision) -> None:
    if backend == "jax":
        configure(jax_precision=precision)
    else:
        configure(tensorflow_precision=precision)


def _evaluate(function, data: dict[str, np.ndarray], backend: str):
    result = function(data)
    if backend == "jax":
        result.block_until_ready()
    return result
