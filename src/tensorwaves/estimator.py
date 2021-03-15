"""Defines estimators which estimate a model's ability to represent the data.

All estimators have to implement the `.Estimator` interface.
"""
from typing import Callable, Dict, Mapping, Union

import numpy as np

from tensorwaves.interfaces import DataSample, Estimator, Model
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
            evaluateable with this dataset.
        phsp_set: A phase space dataset, which is used for the normalization.
            The model has to be evaluateable with this dataset. When correcting
            for the detector efficiency use a phase space sample, that passed
            the detector reconstruction.

    """

    def __init__(
        self,
        model: Model,
        dataset: DataSample,
        phsp_dataset: DataSample,
        phsp_volume: float = 1.0,
        backend: Union[str, tuple, dict] = "numpy",
    ) -> None:
        self.__function = LambdifiedFunction(model, backend)
        self.__gradient = gradient_creator(self.__call__, backend)
        backend_modules = get_backend_modules(backend)

        def find_function_in_backend(name: str) -> Callable:
            if isinstance(backend_modules, dict) and name in backend_modules:
                return backend_modules[name]
            if isinstance(backend_modules, (tuple, list)):
                for module in backend_modules:
                    if name in module.__dict__:
                        return module.__dict__[name]
            raise ValueError(f"Could not find function {name} in backend")

        self.__mean_function = find_function_in_backend("mean")
        self.__sum_function = find_function_in_backend("sum")
        self.__log_function = find_function_in_backend("log")

        self.__dataset = dataset
        self.__dataset = {k: np.array(v) for k, v in dataset.items()}
        self.__phsp_dataset = {k: np.array(v) for k, v in phsp_dataset.items()}
        self.__phsp_volume = phsp_volume

    def __call__(
        self, parameters: Mapping[str, Union[float, complex]]
    ) -> float:
        self.__function.update_parameters(parameters)
        bare_intensities = self.__function(self.__dataset)
        normalization_factor = 1.0 / (
            self.__phsp_volume
            * self.__mean_function(self.__function(self.__phsp_dataset))
        )
        likelihoods = normalization_factor * bare_intensities
        return -self.__sum_function(self.__log_function(likelihoods))

    def gradient(
        self, parameters: Mapping[str, Union[float, complex]]
    ) -> Dict[str, Union[float, complex]]:
        return self.__gradient(parameters)
