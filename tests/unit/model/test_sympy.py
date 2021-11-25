# pylint: disable=invalid-name, no-self-use, redefined-outer-name
import numpy as np
import pytest
import sympy as sp

from tensorwaves.interface import DataSample, Function
from tensorwaves.model import LambdifiedFunction, SympyModel
from tensorwaves.model.sympy import split_expression


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


def create_expression(x, y, z):
    return x ** z + 2 * y


def test_split_expression():
    x, y, z = sp.symbols("x y z")
    expression = create_expression(x, y, z)
    top_expr, sub_expressions = split_expression(expression, max_complexity=3)
    assert top_expr.free_symbols == set(sub_expressions)
    assert expression == top_expr.xreplace(sub_expressions)

    sub_symbols = sorted(top_expr.free_symbols, key=lambda s: s.name)
    assert len(sub_symbols) == 2
    f1, f2 = tuple(sub_symbols)
    assert sub_expressions[f1] == x ** z
    assert sub_expressions[f2] == 2 * y
