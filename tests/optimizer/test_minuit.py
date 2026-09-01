from __future__ import annotations

from typing import TYPE_CHECKING, overload

import pytest

from tensorwaves.interface import (
    Array,
    Estimator,
    FloatArray,
    ParameterType,
    ParameterValue,
)
from tensorwaves.optimizer.minuit import Minuit2

from . import CallbackMock, assert_invocations

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pytest_mock import MockerFixture


class Polynomial1DMinimaEstimator(Estimator):
    def __init__(self, polynomial: Callable) -> None:
        self.__polynomial = polynomial

    @overload
    def __call__(self, parameters: Mapping[str, ParameterValue]) -> float: ...
    @overload
    def __call__(self, parameters: Mapping[str, Array]) -> FloatArray: ...
    @overload
    def __call__(
        self, parameters: Mapping[str, ParameterType]
    ) -> float | FloatArray: ...
    def __call__(self, parameters: Mapping[str, ParameterType]) -> float | FloatArray:
        x = parameters["x"]
        return self.__polynomial(x)

    def gradient(
        self, parameters: Mapping[str, ParameterValue]
    ) -> dict[str, ParameterValue]:
        return NotImplemented


class Polynomial2DMinimaEstimator(Estimator):
    def __init__(self, polynomial: Callable) -> None:
        self.__polynomial = polynomial

    @overload
    def __call__(self, parameters: Mapping[str, ParameterValue]) -> float: ...
    @overload
    def __call__(self, parameters: Mapping[str, Array]) -> FloatArray: ...
    @overload
    def __call__(
        self, parameters: Mapping[str, ParameterType]
    ) -> float | FloatArray: ...
    def __call__(self, parameters: Mapping[str, ParameterType]) -> float | FloatArray:
        x = parameters["x"]
        y = parameters["y"]
        return self.__polynomial(x, y)

    def gradient(
        self, parameters: Mapping[str, ParameterValue]
    ) -> dict[str, ParameterValue]:
        return NotImplemented


class TestMinuit2:
    def test_mock_callback(self, mocker: MockerFixture) -> None:
        estimator = Polynomial1DMinimaEstimator(lambda x: x**2 - 1)
        initial_params = {"x": 0.5}

        callback_stub = mocker.stub(name="callback_stub")
        minuit2 = Minuit2(callback=CallbackMock(callback_stub))
        minuit2.optimize(estimator, initial_params)

        assert_invocations(callback_stub)

    @pytest.mark.parametrize(
        ("estimator", "initial_params", "expected_result"),
        [
            (
                Polynomial1DMinimaEstimator(lambda x: x**2 - 1),
                {"x": 0.5},
                {"x": 0.0},
            ),
            (
                Polynomial1DMinimaEstimator(lambda x: x**2 - 1),
                {"x": -0.5},
                {"x": 0.0},
            ),
            (
                Polynomial1DMinimaEstimator(lambda x: (x - 1) ** 2 - 3 * x + 1),
                {"x": -0.5},
                {"x": 2.5},  # 2 (x - 1) - 3 == 0 -> x = 3/2 + 1
            ),
            (
                Polynomial1DMinimaEstimator(lambda x: x**3 + (x - 1) ** 2 - 3 * x + 1),
                {"x": -1.0},
                {"x": 1.0},
            ),
            (
                Polynomial1DMinimaEstimator(lambda x: x**3 + (x - 1) ** 2 - 3 * x + 1),
                {"x": -2.0},
                None,  # no convergence
            ),
            (
                Polynomial2DMinimaEstimator(lambda x, y: (x - 1) ** 2 + (y + 1) ** 2),
                {"x": -2.0, "y": 4.0},
                {"x": 1.0, "y": -1.0},
            ),
        ],
    )
    def test_optimize(
        self,
        estimator: Estimator,
        initial_params: dict[str, float],
        expected_result: dict[str, float] | None,
    ):
        minuit2 = Minuit2()
        fit_result = minuit2.optimize(estimator, initial_params)

        par_values = fit_result.parameter_values
        par_errors = fit_result.parameter_errors
        assert par_errors is not None

        if expected_result:
            for par_name, value in expected_result.items():
                par_value = par_values[par_name]
                par_error = par_errors[par_name]
                assert isinstance(par_value, float)
                assert isinstance(par_error, float)
                assert value == pytest.approx(par_value, abs=3 * par_error)
        else:
            assert fit_result.minimum_valid is False
