"""Defines estimators which estimate a model's ability to represent the data.

All estimators have to implement the `.Estimator` interface.
"""
from typing import Callable, Dict, Mapping, Union

import numpy as np

from tensorwaves._backend import find_function
from tensorwaves.interface import (
    DataSample,
    Estimator,
    Function,
    ParameterValue,
)


def gradient_creator(
    function: Callable[[Mapping[str, ParameterValue]], ParameterValue],
    backend: Union[str, tuple, dict],
) -> Callable[[Mapping[str, ParameterValue]], Dict[str, ParameterValue]]:
    # pylint: disable=import-outside-toplevel
    if isinstance(backend, str) and backend == "jax":
        import jax
        from jax.config import config

        config.update("jax_enable_x64", True)

        return jax.grad(function)

    def raise_gradient_not_implemented(
        parameters: Mapping[str, ParameterValue]
    ) -> Dict[str, ParameterValue]:
        raise NotImplementedError(
            f"Gradient not implemented for back-end {backend}."
        )

    return raise_gradient_not_implemented


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

    def __init__(  # pylint: disable=too-many-arguments
        self,
        function: Function,
        data: DataSample,
        phsp: DataSample,
        phsp_volume: float = 1.0,
        backend: Union[str, tuple, dict] = "numpy",
    ) -> None:
        self.__data = {k: np.array(v) for k, v in data.items()}
        self.__phsp = {k: np.array(v) for k, v in phsp.items()}
        self.__data_function = function
        self.__phsp_function = function
        self.__gradient = gradient_creator(self.__call__, backend)

        self.__mean_function = find_function(backend, "mean")
        self.__sum_function = find_function(backend, "sum")
        self.__log_function = find_function(backend, "log")

        self.__phsp_volume = phsp_volume

    def __call__(self, parameters: Mapping[str, ParameterValue]) -> float:
        self.__data_function.update_parameters(parameters)
        bare_intensities = self.__data_function(self.__data)
        phsp_intensities = self.__phsp_function(self.__phsp)
        normalization_factor = 1.0 / (
            self.__phsp_volume * self.__mean_function(phsp_intensities)
        )
        likelihoods = normalization_factor * bare_intensities
        return -self.__sum_function(self.__log_function(likelihoods))

    def gradient(
        self, parameters: Mapping[str, ParameterValue]
    ) -> Dict[str, ParameterValue]:
        return self.__gradient(parameters)
