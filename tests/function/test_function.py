# pylint: disable=no-self-use, redefined-outer-name
import numpy as np
import pytest
import sympy as sp

from tensorwaves.function import (
    ParametrizedBackendFunction,
    PositionalArgumentFunction,
)
from tensorwaves.function.sympy import create_parametrized_function
from tensorwaves.interface import DataSample


class TestParametrizedBackendFunction:
    @pytest.fixture(scope="module")
    def function(self) -> ParametrizedBackendFunction:
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
        return create_parametrized_function(
            expression, parameters, backend="numpy"
        )

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
        function,
        test_data: DataSample,
        expected_results: np.ndarray,
    ):
        results = function(test_data)
        np.testing.assert_array_almost_equal(
            results, expected_results, decimal=4
        )


class TestPositionalArgumentFunction:
    def test_call(self):
        function = PositionalArgumentFunction(
            function=lambda a, b, x, y: a * x ** 2 + b * y ** 2,
            argument_order=("a", "b", "x", "y"),
        )
        data: DataSample = {
            "a": np.array([1, 0, +1, 1]),
            "b": np.array([1, 0, -1, 1]),
            "x": np.array([1, 1, +4, 2]),
            "y": np.array([1, 1, -4, 3]),
        }
        output = function(data)
        assert pytest.approx(output) == [2, 0, 0, 4 + 9]
