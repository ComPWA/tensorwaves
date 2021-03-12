# cspell: ignore nfcn

"""Minuit2 adapter to the `iminuit.Minuit` package."""

import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Union

from iminuit import Minuit
from tqdm.auto import tqdm

from tensorwaves.interfaces import Estimator, Optimizer

from .callbacks import Callback, CallbackList


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
                flattened_parameters[par_name] = value  # type: ignore
        return flattened_parameters


class Minuit2(Optimizer):
    """The Minuit2 adapter.

    Implements the `~.interfaces.Optimizer` interface.
    """

    def __init__(
        self,
        callback: Optional[Callback] = None,
        use_analytic_gradient: bool = False,
    ) -> None:
        self.__callback: Callback = CallbackList([])
        if callback is not None:
            self.__callback = callback
        self.__use_gradient = use_analytic_gradient

    def optimize(  # pylint: disable=too-many-locals
        self,
        estimator: Estimator,
        initial_parameters: Mapping[str, Union[complex, float]],
    ) -> Dict[str, Any]:
        parameter_handler = ParameterFlattener(initial_parameters)
        flattened_parameters = parameter_handler.flatten(initial_parameters)

        progress_bar = tqdm(
            disable=logging.getLogger().level > logging.WARNING
        )
        n_function_calls = 0

        def update_parameters(pars: list) -> None:
            for i, k in enumerate(flattened_parameters):
                flattened_parameters[k] = pars[i]

        def wrapped_function(pars: list) -> float:
            nonlocal n_function_calls
            n_function_calls += 1
            update_parameters(pars)
            parameters = parameter_handler.unflatten(flattened_parameters)
            estimator_value = estimator(parameters)
            progress_bar.set_postfix({"estimator": estimator_value})
            progress_bar.update()
            logs = {
                "time": datetime.now(),
                "estimator": {
                    "type": self.__class__.__name__,
                    "value": float(estimator_value),
                },
                "parameters": parameters,
            }
            self.__callback.on_iteration_end(n_function_calls, logs)
            return estimator_value

        def wrapped_gradient(pars: list) -> Iterable[float]:
            update_parameters(pars)
            parameters = parameter_handler.unflatten(flattened_parameters)
            grad = estimator.gradient(parameters)
            return parameter_handler.flatten(grad).values()

        minuit = Minuit(
            wrapped_function,
            tuple(flattened_parameters.values()),
            grad=wrapped_gradient if self.__use_gradient else None,
            name=tuple(flattened_parameters),
        )
        minuit.errors = tuple(
            0.1 * x if x != 0.0 else 0.1 for x in flattened_parameters.values()
        )
        minuit.errordef = (
            Minuit.LIKELIHOOD
        )  # that error definition should be defined in the estimator

        start_time = time.time()
        minuit.migrad()
        end_time = time.time()

        self.__callback.on_function_call_end()

        parameter_values = dict()
        parameter_errors = dict()
        for i, name in enumerate(flattened_parameters):
            par_state = minuit.params[i]
            parameter_values[name] = par_state.value
            parameter_errors[name] = par_state.error

        parameter_values = parameter_handler.unflatten(parameter_values)
        parameter_errors = parameter_handler.unflatten(parameter_errors)

        return {
            "minimum_valid": minuit.valid,
            "parameter_values": parameter_values,
            "parameter_errors": parameter_errors,
            "log_likelihood": minuit.fmin.fval,
            "function_calls": minuit.fmin.nfcn,
            "execution_time": end_time - start_time,
        }
