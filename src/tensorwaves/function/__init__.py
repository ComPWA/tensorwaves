"""Express mathematical expressions in terms of computational functions."""

from typing import Callable, Dict, Mapping, Sequence

import numpy as np

from tensorwaves.interface import (
    DataSample,
    ParameterValue,
    ParametrizedFunction,
)


class ParametrizedBackendFunction(ParametrizedFunction):
    """Implements `.ParametrizedFunction` for a specific computational back-end."""

    def __init__(
        self,
        function: Callable[..., np.ndarray],
        argument_order: Sequence[str],
        parameters: Mapping[str, ParameterValue],
    ) -> None:
        self.__function = function
        self.__argument_order = tuple(argument_order)
        self.__parameters = dict(parameters)

    def __call__(self, dataset: DataSample) -> np.ndarray:
        return self.__function(
            *[
                dataset[var_name]
                if var_name in dataset
                else self.__parameters[var_name]
                for var_name in self.__argument_order
            ],
        )

    @property
    def parameters(self) -> Dict[str, ParameterValue]:
        return dict(self.__parameters)

    def update_parameters(
        self, new_parameters: Mapping[str, ParameterValue]
    ) -> None:
        over_defined = set(new_parameters) - set(self.__parameters)
        if over_defined:
            sep = "\n    "
            parameter_listing = f"{sep}".join(sorted(self.__parameters))
            raise ValueError(
                f"Parameters {over_defined} do not exist in function"
                f" arguments. Expecting one of:{sep}{parameter_listing}"
            )
        self.__parameters.update(new_parameters)
