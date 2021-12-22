"""Defines estimators which estimate a model's ability to represent the data.

All estimators have to implement the `.Estimator` interface.
"""

from typing import Callable, Dict, Mapping, Optional

import numpy as np

from tensorwaves.function._backend import find_function
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


class ChiSquared(Estimator):
    r"""Chi-squared test estimator.

    .. math:: \chi^2 = \sum_{i=1}^n w_i\left(y_i - f_\mathbf{p}(x_i)\right)^2

    Args:
        function: A `.ParametrizedFunction` :math:`f_\mathbf{p}` with
            a set of free `~.ParametrizedFunction.parameters`
            :math:`\mathbf{p}`.
        domain: Input data-set :math:`\mathbf{x}` of :math:`n` events
            :math:`x_i` over which to compute :code:`function`
            :math:`f_\mathbf{p}`.

        observed_values: Observed values :math:`y_i`.
        weights: Optional weights :math:`w_i`. Default: :math:`w_i=1`
            (unweighted). A common choice is :math:`w_i = 1/\sigma_i^2`, with
            :math:`\sigma_i` the uncertainty in each measured value of
            :math:`y_i`.

        backend: Computational backend with which to compute the sum
            :math:`\sum_{i=1}^n`.

    """

    def __init__(
        self,
        function: ParametrizedFunction,
        domain: DataSample,
        observed_values: np.ndarray,
        weights: Optional[np.ndarray] = None,
        backend: str = "numpy",
    ) -> None:
        self.__function = function
        self.__domain = domain
        self.__observed_values = observed_values
        if weights is None:
            ones = find_function("ones", backend)
            self.__weights = ones(len(self.__domain))
        else:
            self.__weights = weights

        self.__gradient = gradient_creator(self.__call__, backend)
        self.__sum = find_function("sum", backend)

    def __call__(self, parameters: Mapping[str, ParameterValue]) -> float:
        self.__function.update_parameters(parameters)
        computed_values = self.__function(self.__domain)
        chi_squared = (
            self.__weights * (computed_values - self.__observed_values) ** 2
        )
        return self.__sum(chi_squared)

    def gradient(
        self, parameters: Mapping[str, ParameterValue]
    ) -> Dict[str, ParameterValue]:
        return self.__gradient(parameters)


class UnbinnedNLL(Estimator):  # pylint: disable=too-many-instance-attributes
    """Unbinned negative log likelihood estimator.

    Args:
        function: A `.ParametrizedFunction` that describes a distribution over
            a certain domain.
        data: The `.DataSample` used for the comparison. The function has to be
            evaluateable with this `.DataSample`.
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
