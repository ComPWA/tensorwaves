# pylint: disable=import-outside-toplevel
import numpy as np
import pytest

from tensorwaves.function._backend import find_function
from tensorwaves.function.sympy import create_parametrized_function


@pytest.mark.parametrize("backend", ["jax", "math", "numba", "numpy", "tf"])
def test_complex_sqrt(backend: str):
    import sympy as sp
    from ampform.dynamics.math import ComplexSqrt
    from numpy.lib.scimath import sqrt as complex_sqrt

    x = sp.Symbol("x")
    expr = ComplexSqrt(x)
    function = create_parametrized_function(
        expr.doit(), parameters={}, backend=backend
    )
    if backend == "math":
        values = -4
    else:
        linspace = find_function("linspace", backend)
        kwargs = {}
        if backend == "tf":
            kwargs["dtype"] = find_function("complex64", backend)
        values = linspace(-4, +4, 9, **kwargs)
    data = {"x": values}
    output_array = function(data)  # type: ignore[arg-type]
    np.testing.assert_almost_equal(output_array, complex_sqrt(data["x"]))
