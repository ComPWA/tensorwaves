from typing import Callable, Dict, Mapping, Optional, Union

import pytest
from pytest_mock import MockerFixture

from tensorwaves.interfaces import Estimator
from tensorwaves.optimizer.scipy import ScipyMinimizer

from . import CallbackMock, assert_invocations


class Polynomial1DMinimaEstimator(Estimator):
    def __init__(self, polynomial: Callable) -> None:
        self.__polynomial = polynomial

    def __call__(
        self, parameters: Mapping[str, Union[float, complex]]
    ) -> float:
        _x = parameters["x"]
        return self.__polynomial(_x)

    def gradient(
        self, parameters: Mapping[str, Union[float, complex]]
    ) -> Dict[str, Union[float, complex]]:
        return NotImplemented


class Polynomial2DMinimaEstimator(Estimator):
    def __init__(self, polynomial: Callable) -> None:
        self.__polynomial = polynomial

    def __call__(
        self, parameters: Mapping[str, Union[float, complex]]
    ) -> float:
        _x = parameters["x"]
        _y = parameters["y"]
        return self.__polynomial(_x, _y)

    def gradient(
        self, parameters: Mapping[str, Union[float, complex]]
    ) -> Dict[str, Union[float, complex]]:
        return NotImplemented


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
            Polynomial1DMinimaEstimator(
                lambda x: x**3 + (x - 1) ** 2 - 3 * x + 1
            ),
            {"x": -1.0},
            {"x": 1.0},
        ),
        (
            Polynomial1DMinimaEstimator(
                lambda x: x**3 + (x - 1) ** 2 - 3 * x + 1
            ),
            {"x": -2.0},
            None,  # no convergence
        ),
        (
            Polynomial2DMinimaEstimator(
                lambda x, y: (x - 1) ** 2 + (y + 1) ** 2
            ),
            {"x": -2.0, "y": 4.0},
            {"x": 1.0, "y": -1.0},
        ),
    ],
)
def test_scipy_optimize(
    estimator: Estimator,
    initial_params: dict,
    expected_result: Optional[dict],
):
    scipy_optimizer = ScipyMinimizer()
    result = scipy_optimizer.optimize(estimator, initial_params)

    par_values = result.parameter_values
    if expected_result:
        assert result.minimum_valid is True
        for par_name, value in expected_result.items():
            assert value == pytest.approx(
                par_values[par_name], rel=1e-2, abs=1e-8
            )
    else:
        assert result.minimum_valid is False


def test_callback(mocker: MockerFixture) -> None:
    estimator = Polynomial1DMinimaEstimator(lambda x: x**2 - 1)
    initial_params = {"x": 0.5}

    callback_stub = mocker.stub(name="callback_stub")
    scipy_optimizer = ScipyMinimizer(callback=CallbackMock(callback_stub))
    scipy_optimizer.optimize(estimator, initial_params)

    assert_invocations(callback_stub)
