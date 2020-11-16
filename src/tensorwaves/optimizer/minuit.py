"""Minuit2 adapter to the `iminuit.Minuit` package."""

import time
from typing import Iterable, Optional

from iminuit import Minuit  # type: ignore

from tensorwaves.interfaces import Estimator, Optimizer

from .callbacks import Callback, CallbackList


class Minuit2(Optimizer):
    """The Minuit2 adapter.

    Implements the `~.interfaces.Optimizer` interface.
    """

    def __init__(self, callbacks: Optional[Iterable[Callback]] = None) -> None:
        self.__callbacks = CallbackList([])
        if callbacks is not None:
            self.__callbacks = CallbackList(callbacks)

    def optimize(self, estimator: Estimator, initial_parameters: dict) -> dict:
        parameters = initial_parameters

        def __call_estimator(params: dict) -> float:
            estimator.update_parameters(params)
            estimator_value = estimator()
            self.__callbacks.call_all(
                parameters=params,
                estimator_value=estimator_value,
            )
            return estimator_value

        def __wrapped_function(pars: list) -> float:
            for i, k in enumerate(parameters.keys()):
                parameters[k] = pars[i]
            return __call_estimator(parameters)

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

        self.__callbacks.finalize_all()

        par_states = minuit.get_param_states()
        f_min = minuit.get_fmin()

        results: dict = {"params": {}}
        for i, name in enumerate(parameters.keys()):
            results["params"][name] = (
                par_states[i].value,
                par_states[i].error,
            )

        # return fit results
        results["log_lh"] = f_min.fval
        results["func_calls"] = f_min.ncalls
        results["time"] = end_time - start_time
        return results
