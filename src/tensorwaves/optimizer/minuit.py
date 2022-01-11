# cspell: ignore nfcn

"""Minuit2 adapter to the `iminuit.Minuit` package."""

import logging
import time
from typing import Iterable, Mapping, Optional

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
    More options can be passed to the internal `~iminuit.Minuit` optimizer by
    calling :meth:`setup` before :meth:`optimize`. This initializes the
    :attr:`iminuit` attribute, so that one can pass options to it before
    calling :meth:`optimize`.
    """

    def __init__(
        self,
        callback: Optional[Callback] = None,
        use_analytic_gradient: bool = False,
    ) -> None:
        self.__callback = callback
        self.__use_gradient = use_analytic_gradient
        self.__iminuit: Optional[iminuit.Minuit] = None

    @property
    def iminuit(self) -> Optional[iminuit.Minuit]:
        """Internal optimizer. Initialize with :meth:`setup`."""
        return self.__iminuit

    def setup(
        self,
        estimator: Estimator,
        initial_parameters: Mapping[str, ParameterValue],
    ) -> None:
        """Initialize internal :attr:`iminuit` optimizer for :meth:`optimize`.

        This sets the internal :attr:`iminuit` instance, so that it can be
        tweaked before calling :meth:`optimize`. See `iminuit.Minuit` for more
        info for available methods.

        .. seealso:: Usage is illustrated under :ref:`usage/basics:Minuit2`.
        """
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

        self.__iminuit = iminuit.Minuit(
            wrapped_function,
            tuple(flattened_parameters.values()),
            grad=wrapped_gradient if self.__use_gradient else None,
            name=tuple(flattened_parameters),
        )
        self.__iminuit.errors = tuple(
            0.1 * x if x != 0.0 else 0.1 for x in flattened_parameters.values()
        )
        self.__iminuit.errordef = (
            iminuit.Minuit.LIKELIHOOD
        )  # that error definition should be defined in the estimator

    def optimize(
        self,
        estimator: Estimator,
        initial_parameters: Mapping[str, ParameterValue],
    ) -> FitResult:
        if self.iminuit is None:
            self.setup(estimator, initial_parameters)
            assert self.iminuit is not None

        start_time = time.time()
        self.iminuit.migrad()
        end_time = time.time()

        parameter_values = {}
        parameter_errors = {}
        for i, name in enumerate(self.iminuit.parameters):
            par_state = self.iminuit.params[i]
            parameter_values[name] = par_state.value
            parameter_errors[name] = par_state.error

        parameter_handler = ParameterFlattener(initial_parameters)
        fit_result = FitResult(
            minimum_valid=self.iminuit.valid,
            execution_time=end_time - start_time,
            function_calls=self.iminuit.fmin.nfcn,
            estimator_value=self.iminuit.fmin.fval,
            parameter_values=parameter_handler.unflatten(parameter_values),
            parameter_errors=parameter_handler.unflatten(parameter_errors),
            specifics=self.iminuit,
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
