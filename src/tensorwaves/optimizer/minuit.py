"""Minuit2 adapter to the `iminuit.Minuit` package."""

import time
from copy import deepcopy
from datetime import datetime
from typing import Dict, Optional

from iminuit import Minuit
from tqdm import tqdm

from tensorwaves.interfaces import Estimator, Optimizer

from .callbacks import Callback, CallbackList


class Minuit2(Optimizer):
    """The Minuit2 adapter.

    Implements the `~.interfaces.Optimizer` interface.
    """

    def __init__(self, callback: Optional[Callback] = None) -> None:
        self.__callback: Callback = CallbackList([])
        if callback is not None:
            self.__callback = callback

    def optimize(
        self, estimator: Estimator, initial_parameters: Dict[str, float]
    ) -> dict:
        parameters = deepcopy(initial_parameters)
        progress_bar = tqdm()
        n_function_calls = 0

        def wrapped_function(pars: list) -> float:
            nonlocal n_function_calls
            n_function_calls += 1
            for i, k in enumerate(parameters.keys()):
                parameters[k] = pars[i]
            estimator.update_parameters(parameters)
            estimator_value = estimator()
            progress_bar.set_postfix({"estimator": estimator_value})
            progress_bar.update()
            logs = {
                "time": datetime.now(),
                "estimator": {
                    "type": self.__class__.__name__,
                    "value": float(estimator_value),
                },
                "parameters": {
                    name: float(value) for name, value in parameters.items()
                },
            }
            self.__callback.on_iteration_end(n_function_calls, logs)
            return estimator_value

        minuit = Minuit.from_array_func(
            wrapped_function,
            list(parameters.values()),
            error=[0.1 * x if x != 0.0 else 0.1 for x in parameters.values()],
            name=list(parameters.keys()),
            errordef=0.5,
        )

        start_time = time.time()
        minuit.migrad()
        end_time = time.time()

        self.__callback.on_function_call_end()

        parameter_values = dict()
        parameter_errors = dict()
        for i, name in enumerate(parameters.keys()):
            par_state = minuit.params[i]
            parameter_values[name] = par_state.value
            parameter_errors[name] = par_state.error

        return {
            "parameter_values": parameter_values,
            "parameter_errors": parameter_errors,
            "log_likelihood": minuit.fmin.fval,
            "function_calls": minuit.fmin.ncalls,
            "execution_time": end_time - start_time,
        }
