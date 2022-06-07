# pylint: disable=redefined-outer-name
from textwrap import dedent

import numpy as np
import pytest
import sympy as sp

from tensorwaves.function import (
    ParametrizedBackendFunction,
    PositionalArgumentFunction,
    get_source_code,
)
from tensorwaves.function.sympy import create_parametrized_function
from tensorwaves.interface import DataSample


class TestParametrizedBackendFunction:
    @pytest.fixture(scope="module")
    def function(self) -> ParametrizedBackendFunction:
        c_1, c_2, c_3, c_4 = sp.symbols("c_(1:5)")
        x = sp.Symbol("x")
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
            + c_3 * (x**2 - 3 * x)
            + c_4
        )
        expression = sp.simplify(sp.conjugate(expression) * expression)
        return create_parametrized_function(
            expression, parameters, backend="numpy"
        )

    def test_argument_order(self, function: ParametrizedBackendFunction):
        assert function.argument_order == ("c_1", "c_2", "c_3", "c_4", "x")

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

    def test_function(self, function: ParametrizedBackendFunction):
        assert callable(function.function)

    def test_update_parameter(self):
        initial_parameter_values = {"a": 1, "b": 1}
        func = ParametrizedBackendFunction(
            lambda a, b, x: a * x + b,
            argument_order=("a", "b", "x"),
            parameters=initial_parameter_values,
        )
        with pytest.raises(
            ValueError,
            match=r"^Parameters {'c'} do not exist in function arguments\.",
        ):
            func.update_parameters({"a": 2, "c": 1})
        assert func.parameters == initial_parameter_values
        new_parameter_values = {"a": 2, "b": 2}
        func.update_parameters(new_parameter_values)
        assert func.parameters == new_parameter_values
        assert new_parameter_values != initial_parameter_values


class TestPositionalArgumentFunction:
    def test_all_unique(self):
        with pytest.raises(
            ValueError, match=r"^There are duplicate argument names: \['b'\]$"
        ):
            PositionalArgumentFunction(
                function=lambda a, b, c: a + b + c,
                argument_order=("a", "b", "b"),
            )

    def test_call(self):
        function = PositionalArgumentFunction(
            function=lambda a, b, x, y: a * x**2 + b * y**2,
            argument_order=("a", "b", "x", "y"),
        )
        assert callable(function.function)
        data: DataSample = {
            "a": np.array([1, 0, +1, 1]),
            "b": np.array([1, 0, -1, 1]),
            "x": np.array([1, 1, +4, 2]),
            "y": np.array([1, 1, -4, 3]),
        }
        output = function(data)
        assert pytest.approx(output) == [2, 0, 0, 4 + 9]

    def test_variadic_args(self):
        function = PositionalArgumentFunction(
            function=lambda *args: args[0] + args[1],
            argument_order=("a", "b"),
        )
        assert callable(function.function)
        data: DataSample = {
            "a": np.array([1, 2, 3]),
            "b": np.array([1, 2, 3]),
        }
        output = function(data)
        assert pytest.approx(output) == [2, 4, 6]


def test_get_source_code():
    def inline_function(a, x):
        return a * x

    function = PositionalArgumentFunction(
        function=inline_function,
        argument_order=("a", "x"),
    )
    src = get_source_code(function)
    expected_src = """
        def inline_function(a, x):
            return a * x
    """
    assert dedent(src).strip() == dedent(expected_src).strip()
