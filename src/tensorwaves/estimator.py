"""Defines estimators which estimate a model's ability to represent the data.

All estimators have to implement the `.Estimator` interface.
"""

from typing import Callable, Dict, Mapping

import numpy as np

from tensorwaves._backend import find_function
from tensorwaves.interface import (
    DataSample,
    Estimator,
    ParameterValue,
    ParametrizedFunction,
)


def gradient_creator(
    function: Callable[[Mapping[str, ParameterValue]], ParameterValue],
    backend: str,
) -> Callable[[Mapping[str, ParameterValue]], Dict[str, ParameterValue]]:
    # pylint: disable=import-outside-toplevel
    if backend == "jax":
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
        function: A `.ParametrizedFunction` that describes a distribution over
            a certain domain.
        data: The dataset used for the comparison. The model has to be
            evaluateable with this dataset.
        phsp: The domain (phase space) over which to execute the function is
            used for the normalization. When correcting for the detector
            efficiency, use a phase space sample that passed the detector
            reconstruction.
        backend: The computational back-end with which the negative log
            likelihood should be computed.

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        function: ParametrizedFunction,
        data: DataSample,
        phsp: DataSample,
        phsp_volume: float = 1.0,
        backend: str = "numpy",
    ) -> None:
        self.__data = {k: np.array(v) for k, v in data.items()}
        self.__phsp = {k: np.array(v) for k, v in phsp.items()}
        self.__function = function
        self.__gradient = gradient_creator(self.__call__, backend)

        self.__mean_function = find_function("mean", backend)
        self.__sum_function = find_function("sum", backend)
        self.__log_function = find_function("log", backend)

        self.__phsp_volume = phsp_volume

    def __call__(self, parameters: Mapping[str, ParameterValue]) -> float:
        self.__function.update_parameters(parameters)
        bare_intensities = self.__function(self.__data)
        phsp_intensities = self.__function(self.__phsp)
        normalization_factor = 1.0 / (
            self.__phsp_volume * self.__mean_function(phsp_intensities)
        )
        likelihoods = normalization_factor * bare_intensities
        return -self.__sum_function(self.__log_function(likelihoods))

    def gradient(
        self, parameters: Mapping[str, ParameterValue]
    ) -> Dict[str, ParameterValue]:
        return self.__gradient(parameters)
