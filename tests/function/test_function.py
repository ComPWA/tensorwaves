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
            + c_2 * sp.exp(-sp.Rational(1, 2) * ((x - 2) / sp.Rational(1, 2)) ** 2)
            + c_3 * (x**2 - 3 * x)
            + c_4
        )
        expression = sp.simplify(sp.conjugate(expression) * expression)
        return create_parametrized_function(expression, parameters, backend="numpy")

    def test_argument_order(self, function: ParametrizedBackendFunction):
        """Test whether data arguments come before parameters."""
        assert function.argument_order == ("x", "c_1", "c_2", "c_3", "c_4")

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
        np.testing.assert_array_almost_equal(results, expected_results, decimal=4)

    def test_function(self, function: ParametrizedBackendFunction):
        assert callable(function.function)

    def test_call_with_parameters(self):
        initial_parameter_values = {"a": 1.0, "b": 2.0}
        func = ParametrizedBackendFunction(
            lambda a, b, x: a * x + b,
            argument_order=("a", "b", "x"),
            parameters=initial_parameter_values,
        )
        data: DataSample = {"x": np.array([0.0, 1.0, 2.0])}
        np.testing.assert_array_equal(func(data), [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(func(data, {"a": -1.0}), [2.0, 1.0, 0.0])
        with pytest.raises(
            ValueError,
            match=r"^Parameters {'c'} do not exist in function arguments\.",
        ):
            func(data, {"a": 2.0, "c": 1.0})
        assert func.parameters == initial_parameter_values
        np.testing.assert_array_equal(func(data), [2.0, 3.0, 4.0])

    def test_with_parameters(self):
        initial_parameter_values = {"a": 1.0, "b": 2.0}
        func = ParametrizedBackendFunction(
            lambda a, b, x: a * x + b,
            argument_order=("a", "b", "x"),
            parameters=initial_parameter_values,
        )
        new_func = func.with_parameters({"a": 2.0})
        assert new_func is not func
        assert new_func.parameters == {"a": 2.0, "b": 2.0}
        assert new_func.function is func.function
        assert func.parameters == initial_parameter_values
        data: DataSample = {"x": np.array([0.0, 1.0, 2.0])}
        np.testing.assert_array_equal(new_func(data), [2.0, 4.0, 6.0])
        with pytest.raises(
            ValueError,
            match=r"^Parameters {'c'} do not exist in function arguments\.",
        ):
            func.with_parameters({"c": 1.0})


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
    def inline_function(a, x):  # ruff:ignore[reimplemented-operator]
        return a * x

    function = PositionalArgumentFunction(
        function=inline_function,
        argument_order=("a", "x"),
    )
    src = get_source_code(function)
    expected_src = """
        def inline_function(a, x):  # ruff:ignore[reimplemented-operator]
            return a * x
    """
    assert dedent(src).strip() == dedent(expected_src).strip()
