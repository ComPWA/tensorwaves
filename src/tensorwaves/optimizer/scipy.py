# cspell:ignore BFGS disp nfev
"""Adapter to the `scipy.optimize` package."""

import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Union

from scipy.optimize import minimize
from tqdm.auto import tqdm

from tensorwaves.interfaces import Estimator, Optimizer

from ._parameter import ParameterFlattener
from .callbacks import Callback, CallbackList


class ScipyMinimizer(Optimizer):
    """The Scipy Optimizer adapter.

    Implements the `~.interfaces.Optimizer` interface.
    """

    def __init__(
        self,
        method: str = "BFGS",
        callback: Optional[Callback] = None,
        use_analytic_gradient: bool = False,
        **scipy_options: Dict[Any, Any],
    ) -> None:
        self.__callback: Callback = CallbackList([])
        if callback is not None:
            self.__callback = callback
        self.__use_gradient = use_analytic_gradient
        self.__method = method
        self.__minimize_options = scipy_options

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
            return list(parameter_handler.flatten(grad).values())

        start_time = time.time()
        res = minimize(
            wrapped_function,
            list(flattened_parameters.values()),
            method=self.__method,
            jac=wrapped_gradient if self.__use_gradient else None,
            options=self.__minimize_options,
        )
        end_time = time.time()

        self.__callback.on_function_call_end()

        parameter_values = parameter_handler.unflatten(
            {
                par_name: res.x[i]
                for i, par_name in enumerate(flattened_parameters)
            }
        )
        return {
            "minimum_valid": res.success,
            "parameter_values": parameter_values,
            "log_likelihood": res.fun,
            "function_calls": res.nfev,
            "execution_time": end_time - start_time,
        }
