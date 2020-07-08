"""Minuit2 adapter to the `iminuit.Minuit` package."""

import logging
import time

from iminuit import Minuit  # type: ignore

from tensorwaves.interfaces import Estimator, Optimizer


class Minuit2(Optimizer):
    """The Minuit2 adapter.

    Implements the `~.interfaces.Optimizer` interface.
    """

    def __init__(self) -> None:
        pass

    def optimize(self, estimator: Estimator, initial_parameters: dict) -> dict:
        parameters = initial_parameters

        function_calls = 0

        def __func(pars: list) -> float:
            """Wrap the estimator."""
            for i, k in enumerate(parameters.keys()):
                parameters[k] = pars[i]
            estimator.update_parameters(parameters)
            nonlocal function_calls
            function_calls += 1
            estimator_val = estimator()
            if function_calls % 10 == 0:
                logging.info(
                    "Function calls: %s\n"
                    "Current estimator value: %s\n"
                    "Parameters: %s",
                    function_calls,
                    estimator_val,
                    list(parameters.values()),
                )
            return estimator_val

        minuit = Minuit.from_array_func(
            __func,
            list(parameters.values()),
            error=[0.1 * x if x != 0.0 else 0.1 for x in parameters.values()],
            name=list(parameters.keys()),
            errordef=0.5,
        )

        start_time = time.time()
        minuit.migrad()
        end_time = time.time()

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
        results["iterations"] = f_min.ncalls
        results["func_calls"] = function_calls
        results["time"] = end_time - start_time
        return results
