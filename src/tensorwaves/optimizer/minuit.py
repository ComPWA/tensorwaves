# cspell: ignore nfcn
"""Minuit2 adapter to the `iminuit.Minuit` package."""
from __future__ import annotations

import logging
import time
from typing import Callable, Iterable, Mapping

import iminuit
from tqdm.auto import tqdm

from tensorwaves.interface import (
    Estimator,
    FitResult,
    Optimizer,
    ParameterValue,
)

from ._parameter import ParameterFlattener
from .callbacks import Callback, _create_log


class Minuit2(Optimizer):
    """Adapter to `Minuit2 <https://root.cern.ch/doc/master/Minuit2Page.html>`_.

    Implements the `~.interface.Optimizer` interface using `iminuit.Minuit`.

    Args:
        callback: Optionally insert behavior through :mod:`.callbacks` into
            the :meth:`optimize` method.
        use_analytic_gradient: Use the :meth:`.Estimator.gradient` when
            calling :meth:`optimize`.
        minuit_modifier: Modify the internal `iminuit.Minuit` optimizer that
            is constructed during the :meth:`optimize` call. See
            :ref:`usage/basics:Minuit2` for an example.
    """

    def __init__(
        self,
        callback: Callback | None = None,
        use_analytic_gradient: bool = False,
        minuit_modifier: Callable[[iminuit.Minuit], None] | None = None,
    ) -> None:
        self.__callback = callback
        self.__use_gradient = use_analytic_gradient
        if minuit_modifier is not None and not callable(minuit_modifier):
            raise TypeError(
                "minuit_modifier has to be a callable that takes a"
                f" {iminuit.Minuit.__module__}.{iminuit.Minuit.__name__} "
                "instance. See constructor signature."
            )
        self.__minuit_modifier = minuit_modifier

    def optimize(  # pylint: disable=too-many-locals
        self,
        estimator: Estimator,
        initial_parameters: Mapping[str, ParameterValue],
    ) -> FitResult:
        parameter_handler = ParameterFlattener(initial_parameters)
        flattened_parameters = parameter_handler.flatten(initial_parameters)

        progress_bar = tqdm(
            disable=logging.getLogger().level > logging.WARNING
        )
        n_function_calls = 0

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

        def wrapped_function(pars: list) -> float:
            nonlocal n_function_calls
            n_function_calls += 1
            update_parameters(pars)
            parameters = parameter_handler.unflatten(flattened_parameters)
            estimator_value = float(estimator(parameters))
            progress_bar.set_postfix({"estimator": estimator_value})
            progress_bar.update()
            if self.__callback is not None:
                self.__callback.on_function_call_end(
                    n_function_calls,
                    logs=_create_log(
                        optimizer=type(self),
                        estimator_type=type(estimator),
                        estimator_value=estimator_value,
                        function_call=n_function_calls,
                        parameters=parameters,
                    ),
                )
            return estimator_value

        def wrapped_gradient(pars: list) -> Iterable[float]:
            update_parameters(pars)
            parameters = parameter_handler.unflatten(flattened_parameters)
            grad = estimator.gradient(parameters)
            return parameter_handler.flatten(grad).values()

        minuit = iminuit.Minuit(
            wrapped_function,
            tuple(flattened_parameters.values()),
            grad=wrapped_gradient if self.__use_gradient else None,
            name=tuple(flattened_parameters),
        )
        minuit.errors = tuple(
            0.1 * x if x != 0.0 else 0.1 for x in flattened_parameters.values()
        )
        minuit.errordef = (
            iminuit.Minuit.LIKELIHOOD
        )  # that error definition should be defined in the estimator

        if self.__minuit_modifier is not None:
            self.__minuit_modifier(minuit)

        start_time = time.time()
        minuit.migrad()
        end_time = time.time()

        parameter_values = {}
        parameter_errors = {}
        for i, name in enumerate(flattened_parameters):
            par_state = minuit.params[i]
            parameter_values[name] = par_state.value
            parameter_errors[name] = par_state.error

        fit_result = FitResult(
            minimum_valid=minuit.valid,
            execution_time=end_time - start_time,
            function_calls=minuit.fmin.nfcn,
            estimator_value=minuit.fmin.fval,
            parameter_values=parameter_handler.unflatten(parameter_values),
            parameter_errors=parameter_handler.unflatten(parameter_errors),
            specifics=minuit,
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
