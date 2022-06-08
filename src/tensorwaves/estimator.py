"""Defines estimators which estimate a model's ability to represent the data.

All estimators have to implement the `.Estimator` interface.
"""
from typing import Callable, Dict, Mapping, Optional, Union

import numpy as np

from tensorwaves.interfaces import DataSample, Estimator, Function, Model
from tensorwaves.model import LambdifiedFunction, get_backend_modules


def gradient_creator(
    function: Callable[[Mapping[str, Union[float, complex]]], float],
    backend: Union[str, tuple, dict],
) -> Callable[
    [Mapping[str, Union[float, complex]]], Dict[str, Union[float, complex]]
]:
    # pylint: disable=import-outside-toplevel
    def not_implemented(
        parameters: Mapping[str, Union[float, complex]]
    ) -> Dict[str, Union[float, complex]]:
        raise NotImplementedError("Gradient not implemented.")

    if isinstance(backend, str) and backend == "jax":
        import jax
        from jax.config import config

        config.update("jax_enable_x64", True)

        return jax.grad(function)

    return not_implemented


class UnbinnedNLL(Estimator):  # pylint: disable=too-many-instance-attributes
    """Unbinned negative log likelihood estimator.

    Args:
        model: A model that should be compared to the dataset.
        dataset: The dataset used for the comparison. The model has to be
            evaluatable with this dataset.
        phsp_set: A phase space dataset, which is used for the normalization.
            The model has to be evaluatable with this dataset. When correcting
            for the detector efficiency use a phase space sample, that passed
            the detector reconstruction.

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model: Union[Function, Model],
        dataset: DataSample,
        phsp_dataset: DataSample,
        phsp_volume: float = 1.0,
        backend: Union[str, tuple, dict] = "numpy",
        use_caching: bool = False,
        fixed_parameters: Optional[Dict[str, Union[float, complex]]] = None,
    ) -> None:
        self.__use_caching = use_caching
        self.__dataset = {k: np.array(v) for k, v in dataset.items()}
        self.__phsp_dataset = {k: np.array(v) for k, v in phsp_dataset.items()}
        if isinstance(model, Function):
            self.__data_function = model
            self.__phsp_function = model
        elif isinstance(model, Model):
            if self.__use_caching:
                fixed_data_inputs = dict(self.__dataset)
                fixed_phsp_inputs = dict(self.__phsp_dataset)
                if fixed_parameters:
                    fixed_data_inputs.update(fixed_parameters)
                    fixed_phsp_inputs.update(fixed_parameters)
                self.__data_function = LambdifiedFunction(
                    model.performance_optimize(fix_inputs=fixed_data_inputs),
                    backend,
                )
                self.__phsp_function = LambdifiedFunction(
                    model.performance_optimize(fix_inputs=fixed_phsp_inputs),
                    backend,
                )
            else:
                self.__data_function = LambdifiedFunction(model, backend)
                self.__phsp_function = self.__data_function
        else:
            raise TypeError(
                f"{model.__class__} not of type {Function} or {Model}"
            )
        self.__gradient = gradient_creator(self.__call__, backend)

        self.__mean_function = _find_function_in_backend(backend, "mean")
        self.__sum_function = _find_function_in_backend(backend, "sum")
        self.__log_function = _find_function_in_backend(backend, "log")

        self.__phsp_volume = phsp_volume

    def __call__(
        self, parameters: Mapping[str, Union[float, complex]]
    ) -> float:
        self.__data_function.update_parameters(parameters)
        if self.__use_caching:
            self.__phsp_function.update_parameters(parameters)
            bare_intensities = self.__data_function({})
            phsp_intensities = self.__phsp_function({})
        else:
            bare_intensities = self.__data_function(self.__dataset)
            phsp_intensities = self.__phsp_function(self.__phsp_dataset)
        normalization_factor = 1.0 / (
            self.__phsp_volume * self.__mean_function(phsp_intensities)
        )
        likelihoods = normalization_factor * bare_intensities
        return -self.__sum_function(self.__log_function(likelihoods))

    def gradient(
        self, parameters: Mapping[str, Union[float, complex]]
    ) -> Dict[str, Union[float, complex]]:
        return self.__gradient(parameters)


def _find_function_in_backend(
    backend: Union[str, tuple, dict], function_name: str
) -> Callable:
    backend_modules = get_backend_modules(backend)
    if isinstance(backend_modules, dict) and function_name in backend_modules:
        return backend_modules[function_name]
    if isinstance(backend_modules, (tuple, list)):
        for module in backend_modules:
            if function_name in module.__dict__:
                return module.__dict__[function_name]
    raise ValueError(f"Could not find function {function_name} in backend")
