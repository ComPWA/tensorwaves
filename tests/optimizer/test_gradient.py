from __future__ import annotations

from itertools import product
from typing import Mapping

import numpy as np
import pytest

from tensorwaves.estimator import gradient_creator
from tensorwaves.interface import ParameterValue


class Function1D:
    def __init__(self, a: float, b: float, c: float) -> None:
        self.__a = a
        self.__b = b
        self.__c = c

    def __call__(
        self, parameters: Mapping[str, ParameterValue]
    ) -> ParameterValue:
        x = parameters["x"]
        return self.__a * x * x + self.__b * x + self.__c

    def true_gradient(
        self, parameters: dict[str, ParameterValue]
    ) -> dict[str, ParameterValue]:
        return {"x": 2.0 * self.__a * parameters["x"] + self.__b}


class Function2D:
    def __init__(self, a: float, b: float, c: float) -> None:
        self.__a = a
        self.__b = b
        self.__c = c

    def __call__(
        self, parameters: Mapping[str, ParameterValue]
    ) -> ParameterValue:
        x = parameters["x"]
        y = parameters["y"]  # pylint: disable=invalid-name
        return self.__a * x * x - self.__b * x * y + self.__c * y

    def true_gradient(
        self, parameters: dict[str, ParameterValue]
    ) -> dict[str, ParameterValue]:
        return {
            "x": 2.0 * self.__a * parameters["x"] - self.__b * parameters["y"],
            "y": -self.__b * parameters["x"] + self.__c,
        }


# Now we just evaluate the gradient function at different positions x and
# compare with the expected values
@pytest.mark.parametrize(
    ("function", "params_cases"),
    [
        (
            Function1D(a=2, b=3, c=5),
            [{"x": x} for x in np.arange(-1.0, 1.0, 0.5)],
        ),
        (
            Function1D(a=-4, b=1, c=2),
            [{"x": x} for x in np.arange(-1.0, 1.0, 0.5)],
        ),
        (
            Function1D(a=3, b=-2, c=-7),
            [{"x": x} for x in np.arange(-1.0, 1.0, 0.5)],
        ),
        (
            Function2D(a=2, b=3, c=5),
            [
                {"x": x, "y": y}
                for x, y in product(
                    np.arange(-1.0, 1.0, 0.5), np.arange(-1.0, 1.0, 0.5)
                )
            ],
        ),
        (
            Function2D(a=-4, b=1, c=2),
            [
                {"x": x, "y": y}
                for x, y in product(
                    np.arange(-1.0, 1.0, 0.5), np.arange(-1.0, 1.0, 0.5)
                )
            ],
        ),
        (
            Function2D(a=3, b=-2, c=-7),
            [
                {"x": x, "y": y}
                for x, y in product(
                    np.arange(-1.0, 1.0, 0.5), np.arange(-1.0, 1.0, 0.5)
                )
            ],
        ),
    ],
)
def test_jax_gradient(
    function: Function1D | Function2D,
    params_cases: list[dict[str, ParameterValue]],
):
    gradient = gradient_creator(function, backend="jax")
    for params in params_cases:
        gradient_values = gradient(params)
        assert gradient_values == function.true_gradient(params)
