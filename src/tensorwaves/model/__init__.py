"""Evaluateable physics models for amplitude analysis.

The `.model` module takes care of lambdifying mathematical expressions to
computational backends.
"""

# pyright: reportUnusedImport=false
from typing import Dict, Mapping, Union

import numpy as np

from tensorwaves.interface import DataSample, Function, Model, ParameterValue

from .backend import get_backend_modules  # noqa: F401
from .sympy import SympyModel  # noqa: F401


class LambdifiedFunction(Function):
    """Implements `.Function` based on a `.Model` using {meth}`~.Model.lambdify`."""

    def __init__(
        self,
        model: Model,
        backend: Union[str, tuple, dict] = "numpy",
    ) -> None:
        self.__lambdified_model = model.lambdify(backend=backend)
        self.__parameters = model.parameters
        self.__ordered_args = model.argument_order

    def __call__(self, dataset: DataSample) -> np.ndarray:
        return self.__lambdified_model(
            *[
                dataset[var_name]
                if var_name in dataset
                else self.__parameters[var_name]
                for var_name in self.__ordered_args
            ],
        )

    @property
    def parameters(self) -> Dict[str, ParameterValue]:
        return self.__parameters

    def update_parameters(
        self, new_parameters: Mapping[str, ParameterValue]
    ) -> None:
        if not set(new_parameters) <= set(self.__parameters):
            over_defined = set(new_parameters) ^ set(self.__parameters)
            raise ValueError(
                f"Parameters {over_defined} do not exist in function arguments"
            )
        self.__parameters.update(new_parameters)
