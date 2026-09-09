from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tensorwaves.optimizer.minuit import Minuit2

from . import (
    POLYNOMIAL_MINIMA_CASES,
    CallbackMock,
    Polynomial1DMinimaEstimator,
    assert_invocations,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from tensorwaves.interface import Estimator


def describe_Minuit2():
    def it_invokes_the_callbacks_in_order(mocker: MockerFixture) -> None:
        estimator = Polynomial1DMinimaEstimator(lambda x: x**2 - 1)
        initial_params = {"x": 0.5}

        callback_stub = mocker.stub(name="callback_stub")
        minuit2 = Minuit2(callback=CallbackMock(callback_stub))
        minuit2.optimize(estimator, initial_params)

        assert_invocations(callback_stub)

    @pytest.mark.parametrize(
        ("estimator", "initial_params", "expected_result"), POLYNOMIAL_MINIMA_CASES
    )
    def it_finds_the_minimum_within_the_parameter_errors(
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
