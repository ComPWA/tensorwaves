# cspell:ignore lambdifygenerated
# pylint: disable=invalid-name, no-self-use, redefined-outer-name
import sys
from typing import Tuple

import numpy as np
import pytest
import sympy as sp

from tensorwaves.interface import DataSample, Function
from tensorwaves.model import LambdifiedFunction, SympyModel
from tensorwaves.model.sympy import optimized_lambdify, split_expression


class TestLambdifiedFunction:
    @pytest.fixture(scope="module")
    def function(self) -> LambdifiedFunction:
        c_1, c_2, c_3, c_4 = sp.symbols("c_(1:5)")
        x = sp.Symbol("x", complex_twice=True)
        parameters = {
            c_1: 1 + 1j,
            c_2: -1 + 1j,
            c_3: 1 - 1j,
            c_4: -1 - 1j,
        }
        expression = (
            c_1 * sp.sqrt(x) / x
            + c_2
            * sp.exp(-sp.Rational(1, 2) * ((x - 2) / sp.Rational(1, 2)) ** 2)
            + c_3 * (x ** 2 - 3 * x)
            + c_4
        )
        expression = sp.simplify(sp.conjugate(expression) * expression)
        model = SympyModel(expression=expression, parameters=parameters)
        return LambdifiedFunction(model, "numpy")

    @pytest.mark.parametrize(
        ("test_data", "expected_results"),
        [
            (
                {"x": np.array([0.5, 1.0, 1.5, 2.0, 2.5])},
                [3.52394, 9.11931, 16.3869, 18.1716, 7.16359],
            ),
        ],
    )
    def test_call(
        self,
        function: Function,
        test_data: DataSample,
        expected_results: np.ndarray,
    ):
        results = function(test_data)
        np.testing.assert_array_almost_equal(
            results, expected_results, decimal=4
        )


def create_expression(a, x, y, z) -> sp.Expr:
    return a * (x ** z + 2 * y)


@pytest.mark.parametrize("backend", ["jax", "math", "numpy", "tf"])
@pytest.mark.parametrize("max_complexity", [0, 1, 2, 3, 4, 5])
def test_optimized_lambdify(backend: str, max_complexity: int):
    symbols: Tuple[sp.Symbol, ...] = sp.symbols("a x y z")
    a, x, y, z = symbols
    expression = create_expression(a, x, y, z)
    function = optimized_lambdify(
        expression,
        symbols,
        backend=backend,
        max_complexity=max_complexity,
    )

    func_repr = str(function)
    if 0 < max_complexity <= 4:
        repr_start = "<function fast_lambdify.<locals>"
    else:
        repr_start = "<function _lambdifygenerated"
        if backend == "jax":
            if sys.version_info >= (3, 7):
                repr_start = "<CompiledFunction of " + repr_start
            else:
                repr_start = "<CompiledFunction object at 0x"
        assert func_repr.startswith(repr_start)

    data = (
        4,
        np.array([1, 2]),
        np.array([1, np.e]),
        np.array([1, 2]),
    )
    output = function(*data)
    expected = create_expression(*data)
    assert pytest.approx(output) == expected


def test_split_expression():
    symbols: Tuple[sp.Symbol, ...] = sp.symbols("a x y z")
    a, x, y, z = symbols
    expression = create_expression(a, x, y, z)

    assert expression.args[0] is a
    assert len(expression.args[1].args) == 2
    sub_expr, _ = expression.args[1].args
    assert sub_expr == x ** z
    n_nodes = sp.count_ops(sub_expr)
    assert n_nodes == 1

    top_expr, sub_expressions = split_expression(
        expression,
        min_complexity=n_nodes,
        max_complexity=n_nodes,
    )
    assert top_expr.free_symbols == set(sub_expressions)
    assert expression == top_expr.xreplace(sub_expressions)

    sub_symbols = sorted(top_expr.free_symbols, key=lambda s: s.name)
    assert len(sub_symbols) == 3
    f0, f1, f2 = tuple(sub_symbols)
    assert f0 is a
    assert sub_expressions[f0] == a
    assert sub_expressions[f1] == x ** z
    assert sub_expressions[f2] == 2 * y
