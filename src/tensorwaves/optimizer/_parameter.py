from typing import Dict, Mapping, Union


class ParameterFlattener:
    def __init__(
        self, parameters: Mapping[str, Union[float, complex]]
    ) -> None:
        self.__real_imag_to_complex_name = {}
        self.__complex_to_real_imag_name = {}
        for name, val in parameters.items():
            if isinstance(val, complex):
                real_name = f"real_{name}"
                imag_name = f"imag_{name}"
                self.__real_imag_to_complex_name[real_name] = name
                self.__real_imag_to_complex_name[imag_name] = name
                self.__complex_to_real_imag_name[name] = (real_name, imag_name)

    def unflatten(
        self, flattened_parameters: Dict[str, float]
    ) -> Dict[str, Union[float, complex]]:
        parameters: Dict[str, Union[float, complex]] = {
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
        self, parameters: Mapping[str, Union[float, complex]]
    ) -> Dict[str, float]:
        flattened_parameters = {}
        for par_name, value in parameters.items():
            if par_name in self.__complex_to_real_imag_name:
                (real_name, imag_name) = self.__complex_to_real_imag_name[
                    par_name
                ]
                flattened_parameters[real_name] = parameters[par_name].real
                flattened_parameters[imag_name] = parameters[par_name].imag
            else:
                flattened_parameters[par_name] = value  # type: ignore[assignment]
        return flattened_parameters
