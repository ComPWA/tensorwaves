from __future__ import annotations

from typing import Mapping

from tensorwaves.interface import ParameterValue


class ParameterFlattener:
    def __init__(self, parameters: Mapping[str, ParameterValue]) -> None:
        self.__real_imag_to_complex_name: dict[str, str] = {}
        self.__complex_to_real_imag_name: dict[str, tuple[str, str]] = {}
        for name, val in parameters.items():
            if isinstance(val, complex):
                real_name = f"real_{name}"
                imag_name = f"imag_{name}"
                self.__real_imag_to_complex_name[real_name] = name
                self.__real_imag_to_complex_name[imag_name] = name
                self.__complex_to_real_imag_name[name] = (real_name, imag_name)

    def unflatten(
        self, flattened_parameters: dict[str, float]
    ) -> dict[str, ParameterValue]:
        parameters: dict[str, ParameterValue] = {
            k: v
            for k, v in flattened_parameters.items()
            if k not in self.__real_imag_to_complex_name
        }
        for complex_name, (
            real_name,
            imag_name,
        ) in self.__complex_to_real_imag_name.items():
            parameters[complex_name] = complex(
                flattened_parameters[real_name],
                flattened_parameters[imag_name],
            )
        return parameters

    def flatten(
        self, parameters: Mapping[str, ParameterValue]
    ) -> dict[str, float]:
        flattened_parameters: dict[str, float] = {}
        for par_name, value in parameters.items():
            if isinstance(value, complex):
                if par_name not in self.__complex_to_real_imag_name:
                    raise ValueError(
                        f"Parameter '{par_name}' has was not registered upon"
                        f" constructing the {type(self).__name__}"
                    )
                name_pair = self.__complex_to_real_imag_name[par_name]
                real_name, imag_name = name_pair
                flattened_parameters[real_name] = value.real
                flattened_parameters[imag_name] = value.imag
            else:
                flattened_parameters[par_name] = value
        return flattened_parameters
