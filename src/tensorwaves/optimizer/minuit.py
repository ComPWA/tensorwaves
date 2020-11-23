"""Minuit2 adapter to the `iminuit.Minuit` package."""

import time
from copy import deepcopy
from typing import Dict, Optional

from iminuit import Minuit

from tensorwaves.interfaces import Estimator, Optimizer

from .callbacks import Callback, CallbackList, progress_bar


class Minuit2(Optimizer):
    """The Minuit2 adapter.

    Implements the `~.interfaces.Optimizer` interface.
    """

    def __init__(self, callback: Optional[Callback] = None) -> None:
        self.__callback: Callback = CallbackList([])
        if callback is not None:
            self.__callback = callback

    @progress_bar()
    def optimize(
        self, estimator: Estimator, initial_parameters: Dict[str, float]
    ) -> dict:
        parameters = deepcopy(initial_parameters)

        def __wrapped_function(pars: list) -> float:
            for i, k in enumerate(parameters.keys()):
                parameters[k] = pars[i]
            estimator.update_parameters(parameters)
            estimator_value = estimator()
            self.__callback(parameters, estimator_value)
            return estimator_value

        minuit = Minuit.from_array_func(
            __wrapped_function,
            list(parameters.values()),
            error=[0.1 * x if x != 0.0 else 0.1 for x in parameters.values()],
            name=list(parameters.keys()),
            errordef=0.5,
        )

        start_time = time.time()
        minuit.migrad()
        end_time = time.time()

        self.__callback.finalize()

        par_states = minuit.params
        f_min = minuit.fmin

        parameter_values = dict()
        parameter_errors = dict()
        for i, name in enumerate(parameters.keys()):
            par_state = par_states[i]
            parameter_values[name] = par_state.value
            parameter_errors[name] = par_state.error

        return {
            "parameter_values": parameter_values,
            "parameter_errors": parameter_errors,
            "log_likelihood": f_min.fval,
            "function_calls": f_min.ncalls,
            "execution_time": end_time - start_time,
        }
