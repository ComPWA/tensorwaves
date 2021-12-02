# pylint: disable=import-outside-toplevel
import numpy as np
import pytest

from tensorwaves.model import LambdifiedFunction, SympyModel
from tensorwaves.model._backend import find_function


@pytest.mark.parametrize("backend", ["jax", "math", "numba", "numpy", "tf"])
def test_complex_sqrt(backend: str):
    import sympy as sp
    from ampform.dynamics.math import ComplexSqrt
    from numpy.lib.scimath import sqrt as complex_sqrt

    x = sp.Symbol("x")
    expr = ComplexSqrt(x)
    model = SympyModel(expr, parameters={})
    function = LambdifiedFunction(model, backend)
    if backend == "math":
        values = -4
    else:
        linspace = find_function(backend, "linspace")
        kwargs = {}
        if backend == "tf":
            kwargs["dtype"] = find_function(backend, "complex64")
        values = linspace(-4, +4, 9, **kwargs)
    data = {"x": values}
    output_array = function(data)  # type: ignore[arg-type]
    np.testing.assert_almost_equal(output_array, complex_sqrt(data["x"]))
