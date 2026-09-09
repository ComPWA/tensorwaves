from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tensorwaves.optimizer.scipy import ScipyMinimizer

from . import (
    POLYNOMIAL_MINIMA_CASES,
    CallbackMock,
    Polynomial1DMinimaEstimator,
    assert_invocations,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from tensorwaves.interface import Estimator


def describe_ScipyMinimizer():
    def it_invokes_the_callbacks_in_order(mocker: MockerFixture) -> None:
        estimator = Polynomial1DMinimaEstimator(lambda x: x**2 - 1)
        initial_params = {"x": 0.5}

        callback_stub = mocker.stub(name="callback_stub")
        scipy_optimizer = ScipyMinimizer(callback=CallbackMock(callback_stub))
        scipy_optimizer.optimize(estimator, initial_params)

        assert_invocations(callback_stub)

    @pytest.mark.parametrize(
        ("estimator", "initial_params", "expected_result"), POLYNOMIAL_MINIMA_CASES
    )
    def it_finds_the_minimum(
        estimator: Estimator,
        initial_params: dict,
        expected_result: dict | None,
    ):
        scipy_optimizer = ScipyMinimizer()
        fit_result = scipy_optimizer.optimize(estimator, initial_params)

        par_values = fit_result.parameter_values
        if expected_result:
            assert fit_result.minimum_valid is True
            for par_name, value in expected_result.items():
                assert value == pytest.approx(par_values[par_name], rel=1e-2, abs=1e-8)
        else:
            assert fit_result.minimum_valid is False
