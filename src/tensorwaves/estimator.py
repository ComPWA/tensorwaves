"""Defines estimators which estimate a model's ability to represent the data.

All estimators have to implement the `.Estimator` interface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, Mapping

import numpy as np

from tensorwaves.data.transform import SympyDataTransformer
from tensorwaves.function._backend import (
    find_function,
    raise_missing_module_error,
)
from tensorwaves.function.sympy import (
    create_parametrized_function,
    prepare_caching,
)
from tensorwaves.interface import (
    DataSample,
    DataTransformer,
    Estimator,
    ParameterValue,
    ParametrizedFunction,
)

if TYPE_CHECKING:
    import sympy as sp


def create_cached_function(
    expression: sp.Expr,
    parameters: Mapping[sp.Symbol, ParameterValue],
    backend: str,
    free_parameters: Iterable[sp.Symbol],
    use_cse: bool = True,
) -> tuple[ParametrizedFunction, DataTransformer]:
    """Create a function and data transformer for cached computations.

    Once it is known which parameters in an expression are to be optimized,
    this function makes it easy to cache constant sub-trees.

    Args:
        expression: The `~sympy.core.expr.Expr` that should be expressed in a
            computational backend.
        parameters: Symbols in the :code:`expression` that should be
            interpreted as parameters. The values in this mapping will be used
            in the returned :attr:`.ParametrizedFunction.parameters`.
        backend: The computational backend to which in which to express the
            input :code:`expression`.

        use_cse: See :func:`.create_parametrized_function`.

    Returns:
        A 'cached' `.ParametrizedFunction` with only the free
        `~.ParametrizedFunction.parameters` that are to be optimized and a
        `.DataTransformer` that needs to be used to transform a data sample
        for the original expresion to the cached function.

    .. seealso:: This function is an extension of :func:`.prepare_caching` and
        :func:`.create_parametrized_function`. :doc:`/usage/caching` shows how
        to use this function.
    """
    cache_expression, transformer_expressions = prepare_caching(
        expression, parameters, free_parameters
    )
    free_parameter_values = {
        par: value
        for par, value in parameters.items()
        if par in free_parameters
    }
    cached_function = create_parametrized_function(
        cache_expression, free_parameter_values, backend, use_cse=use_cse
    )
    cache_transformer = SympyDataTransformer.from_sympy(
        transformer_expressions, backend, use_cse=use_cse
    )
    return cached_function, cache_transformer


def gradient_creator(
    function: Callable[[Mapping[str, ParameterValue]], ParameterValue],
    backend: str,
) -> Callable[[Mapping[str, ParameterValue]], dict[str, ParameterValue]]:
    # pylint: disable=import-outside-toplevel
    if backend == "jax":
        try:
            import jax
            from jax.config import config
        except ImportError:  # pragma: no cover
            raise_missing_module_error("jax", extras_require="jax")

        config.update("jax_enable_x64", True)

        return jax.grad(function)

    def raise_gradient_not_implemented(
        parameters: Mapping[str, ParameterValue]
    ) -> dict[str, ParameterValue]:
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

    .. seealso:: :doc:`/usage/chi-squared`
    """

    def __init__(
        self,
        function: ParametrizedFunction,
        domain: DataSample,
        observed_values: np.ndarray,
        weights: np.ndarray | None = None,
        backend: str = "numpy",
    ) -> None:
        self.__function = function
        self.__domain = domain
        self.__observed_values = observed_values
        if weights is None:
            ones = find_function("ones", backend)
            self.__weights = ones(len(self.__observed_values))
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
    ) -> dict[str, ParameterValue]:
        return self.__gradient(parameters)


class UnbinnedNLL(Estimator):  # pylint: disable=too-many-instance-attributes
    r"""Unbinned negative log likelihood estimator.

    The **log likelihood** :math:`\log\mathcal{L}` for a given function
    :math:`f_\mathbf{p}: X^m \rightarrow \mathbb{R}` over :math:`N` data points
    :math:`\mathbf{x}` and over a (phase space) domain of
    :math:`n_\mathrm{phsp}` points :math:`\mathbf{x}_\mathrm{phsp}`, is given
    by:

    .. math::

        -\log\mathcal{L} = N\log\lambda
        -\sum_{i=1}^N \log\left(f_\mathbf{p}(x_i)\right)

    with :math:`\lambda` the normalization integral over :math:`f_\mathbf{p}`.
    The integral is computed numerically by averaging over a significantly
    large (phase space) domain sample :math:`\mathbf{x}_\mathrm{phsp}` of size
    :math:`n`:

    .. math::
        \lambda = \frac{\sum_{j=1}^n V f_\mathbf{p}(x_{\mathrm{phsp},j})}{n}.

    Args:
        function: A `.ParametrizedFunction` :math:`f_\mathbf{p}` that describes
            a distribution over a certain domain.
        data: The `.DataSample` :math:`\mathbf{x}` over which to compute
            :math:`f_\mathbf{p}`.
        phsp: The domain (phase space) with which the likelihood is normalized.
            When correcting for the detector efficiency, use a phase space
            sample that passed the detector reconstruction.
        phsp_volume: Optional phase space volume :math:`V`, used in the
            normalization factor. Default: :math:`V=1`.
        backend: The computational back-end with which the sums and averages
            should be computed.

    .. seealso:: :doc:`/usage/unbinned-fit`
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
    ) -> dict[str, ParameterValue]:
        return self.__gradient(parameters)
