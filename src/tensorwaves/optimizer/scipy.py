# cspell:ignore BFGS disp nfev
"""Adapter to the `scipy.optimize` package."""
from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Mapping

from tqdm.auto import tqdm

from tensorwaves.function._backend import raise_missing_module_error
from tensorwaves.interface import (
    Estimator,
    FitResult,
    Optimizer,
    ParameterValue,
)

from ._parameter import ParameterFlattener
from .callbacks import Callback, _create_log


class ScipyMinimizer(Optimizer):
    """The Scipy Optimizer adapter.

    Implements the `~.interface.Optimizer` interface.

    .. seealso:: :ref:`usage/basics:Scipy`
    """

    def __init__(
        self,
        method: str = "BFGS",
        callback: Callback | None = None,
        use_analytic_gradient: bool = False,
        **scipy_options: dict[Any, Any],
    ) -> None:
        self.__callback = callback
        self.__use_gradient = use_analytic_gradient
        self.__method = method
        self.__minimize_options = scipy_options

    def optimize(  # pylint: disable=too-many-locals
        self,
        estimator: Estimator,
        initial_parameters: Mapping[str, ParameterValue],
    ) -> FitResult:
        # pylint: disable=import-outside-toplevel
        try:
            from scipy.optimize import minimize
        except ImportError:  # pragma: no cover
            raise_missing_module_error("scipy", extras_require="scipy")

        parameter_handler = ParameterFlattener(initial_parameters)
        flattened_parameters = parameter_handler.flatten(initial_parameters)

        progress_bar = tqdm(
            disable=logging.getLogger().level > logging.WARNING
        )
        n_function_calls = 0
        iterations = 0
        estimator_value = 0.0

        parameters = parameter_handler.unflatten(flattened_parameters)
        if self.__callback is not None:
            self.__callback.on_optimize_start(
                logs=_create_log(
                    optimizer=type(self),
                    estimator_type=type(estimator),
                    estimator_value=estimator(parameters),
                    function_call=n_function_calls,
                    parameters=parameters,
                )
            )

        def update_parameters(pars: list) -> None:
            for i, k in enumerate(flattened_parameters):
                flattened_parameters[k] = pars[i]

        def create_parameter_dict(
            pars: Iterable[float],
        ) -> dict[str, ParameterValue]:
            return parameter_handler.unflatten(
                dict(zip(flattened_parameters.keys(), pars))
            )

        def wrapped_function(pars: list) -> float:
            nonlocal n_function_calls
            nonlocal estimator_value
            n_function_calls += 1
            update_parameters(pars)
            parameters = parameter_handler.unflatten(flattened_parameters)
            estimator_value = estimator(parameters)
            progress_bar.set_postfix({"estimator": estimator_value})
            progress_bar.update()
            if self.__callback is not None:
                self.__callback.on_function_call_end(
                    n_function_calls,
                    logs=_create_log(
                        optimizer=type(self),
                        estimator_type=type(estimator),
                        estimator_value=estimator(parameters),
                        function_call=n_function_calls,
                        parameters=parameters,
                    ),
                )
            return float(estimator_value)

        def wrapped_gradient(pars: list) -> Iterable[float]:
            update_parameters(pars)
            parameters = parameter_handler.unflatten(flattened_parameters)
            grad = estimator.gradient(parameters)
            return list(parameter_handler.flatten(grad).values())

        def wrapped_callback(pars: Iterable[float]) -> None:
            nonlocal iterations
            iterations += 1
            if self.__callback is not None:
                self.__callback.on_iteration_end(
                    iterations,
                    logs=_create_log(
                        optimizer=type(self),
                        estimator_type=type(estimator),
                        estimator_value=float(estimator_value),
                        function_call=n_function_calls,
                        parameters=create_parameter_dict(pars),
                    ),
                )

        start_time = time.time()
        fit_result = minimize(
            wrapped_function,
            list(flattened_parameters.values()),
            method=self.__method,
            jac=wrapped_gradient if self.__use_gradient else None,
            options=self.__minimize_options,
            callback=wrapped_callback,
        )
        end_time = time.time()

        fit_result = FitResult(
            minimum_valid=fit_result.success,
            execution_time=end_time - start_time,
            function_calls=fit_result.nfev,
            estimator_value=fit_result.fun,
            parameter_values=create_parameter_dict(fit_result.x),
            iterations=fit_result.nit,
            specifics=fit_result,
        )
        if self.__callback is not None:
            self.__callback.on_optimize_end(
                logs=_create_log(
                    optimizer=type(self),
                    estimator_type=type(estimator),
                    estimator_value=fit_result.estimator_value,
                    function_call=fit_result.function_calls,
                    parameters=fit_result.parameter_values,
                )
            )
        return fit_result
