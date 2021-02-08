from typing import Callable, Optional

import pytest

from tensorwaves.interfaces import Estimator
from tensorwaves.optimizer.minuit import Minuit2


class Polynomial1DMinimaEstimator:
    def __init__(self, polynomial: Callable) -> None:
        self.__polynomial = polynomial

    def __call__(self, parameters: dict) -> float:
        _x = parameters["x"]
        return self.__polynomial(_x)

    @property
    def parameters(self) -> dict:
        return {"x": 0.0}


class Polynomial2DMinimaEstimator:
    def __init__(self, polynomial: Callable) -> None:
        self.__polynomial = polynomial

    def __call__(self, parameters: dict) -> float:
        _x = parameters["x"]
        _y = parameters["y"]
        return self.__polynomial(_x, _y)

    @property
    def parameters(self) -> dict:
        return {"x": 0.0, "y": 0.0}


@pytest.mark.parametrize(
    "estimator, initial_params, expected_result",
    [
        (
            Polynomial1DMinimaEstimator(lambda x: x ** 2 - 1),
            {"x": 0.5},
            {"x": 0.0},
        ),
        (
            Polynomial1DMinimaEstimator(lambda x: x ** 2 - 1),
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
                lambda x: x ** 3 + (x - 1) ** 2 - 3 * x + 1
            ),
            {"x": -1.0},
            {"x": 1.0},
        ),
        (
            Polynomial1DMinimaEstimator(
                lambda x: x ** 3 + (x - 1) ** 2 - 3 * x + 1
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
def test_minuit2(
    estimator: Estimator, initial_params: dict, expected_result: Optional[dict]
):
    minuit2 = Minuit2()
    result = minuit2.optimize(estimator, initial_params)

    par_values = result["parameter_values"]
    par_errors = result["parameter_errors"]

    if expected_result:
        for par_name, value in expected_result.items():
            assert value == pytest.approx(
                par_values[par_name], abs=3 * par_errors[par_name]
            )
    else:
        assert result["minimum_valid"] is False
