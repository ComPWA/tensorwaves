"""Defines estimators which estimate a model's ability to represent the data.

All estimators have to implement the `.Estimator` interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from tensorwaves.config import _initialize_jax
from tensorwaves.data.transform import SympyDataTransformer
from tensorwaves.function._backend import find_function, raise_missing_module_error
from tensorwaves.function.sympy import create_parametrized_function, prepare_caching
from tensorwaves.interface import (
    Array,
    DataSample,
    DataTransformer,
    Estimator,
    FloatArray,
    ParameterType,
    ParameterValue,
    ParametrizedFunction,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    import sympy as sp


def create_cached_function(
    expression: sp.Expr,
    parameters: Mapping[sp.Basic, ParameterValue],
    backend: str,
    free_parameters: Iterable[sp.Basic],
    *,
    use_cse: bool = True,
) -> tuple[ParametrizedFunction[DataSample, Array], DataTransformer]:
    """Create a function and data transformer for cached computations.

    Once it is known which parameters in an expression are to be optimized, this
    function makes it easy to cache constant sub-trees.

    Args:
        expression: The `~sympy.core.expr.Expr` that should be expressed in a
            computational backend.
        parameters: Symbols in the :code:`expression` that should be
            interpreted as parameters. The values in this mapping will be used in the
            returned :attr:`.ParametrizedFunction.parameters`.
        backend: The computational backend to which in which to express the
            input :code:`expression`.
        free_parameters: Symbols in the expression that change and should not be cached.
        use_cse: See :func:`.create_parametrized_function`.

    Returns:
        A 'cached' `.ParametrizedFunction` with only the free
        `~.ParametrizedFunction.parameters` that are to be optimized and a
        `.DataTransformer` that needs to be used to transform a data sample for the
        original expresion to the cached function.

    .. seealso:: This function is an extension of :func:`.prepare_caching` and
        :func:`.create_parametrized_function`. :doc:`/usage/caching` shows how
        to use this function.
    """
    cache_expression, transformer_expressions = prepare_caching(
        expression, parameters, free_parameters
    )
    free_parameter_values = {
        par: value for par, value in parameters.items() if par in free_parameters
    }
    cached_function = create_parametrized_function(
        cache_expression, free_parameter_values, backend, use_cse=use_cse
    )
    cache_transformer = SympyDataTransformer.from_sympy(
        transformer_expressions, backend, use_cse=use_cse
    )
    return cached_function, cache_transformer


def _determine_backend(function: ParametrizedFunction, backend: str | None) -> str:
    if backend is not None:
        return backend
    function_backend = getattr(function, "backend", None)
    if function_backend is None:
        return "numpy"
    return function_backend


def _coerce_parameter_types(
    parameters: Mapping[str, ParameterType],
) -> dict[str, ParameterType]:
    return {name: _coerce_parameter_value(value) for name, value in parameters.items()}


def _coerce_parameter_value(value: ParameterType) -> ParameterType:
    """Normalize values for stable JIT inputs and event-axis broadcasting.

    Scalars are converted to ``float`` or ``complex`` so that a change from an integer
    value does not trigger retracing. Parameter arrays receive a new trailing axis so
    that they broadcast against the data samples' event axis.
    """
    if isinstance(value, complex):
        return complex(value)
    if isinstance(value, (int, float)):
        return float(value)
    if getattr(value, "ndim", 0) >= 1:
        return value[..., None]
    return value


def _import_jax():  # ruff: ignore[missing-return-type-private-function]
    try:
        return _initialize_jax()
    except ImportError:  # pragma: no cover
        raise_missing_module_error("jax", extras_require="jax")


def _conjugate_complex_gradient(
    gradient: Mapping[str, ParameterValue],
) -> dict[str, ParameterValue]:
    # jax.grad() returns the conjugated Wirtinger derivative ∂f/∂x - i∂f/∂y
    # for complex-valued parameters, so conjugate to get a complex number
    # whose real and imaginary parts are (∂f/∂x, ∂f/∂y)
    return {name: value.conjugate() for name, value in gradient.items()}


def gradient_creator(
    function: Callable[[Mapping[str, ParameterValue]], ParameterValue],
    backend: str,
) -> Callable[[Mapping[str, ParameterValue]], dict[str, ParameterValue]]:
    if backend == "jax":
        jax = _import_jax()
        gradient = jax.grad(function)

        def conjugated_gradient(
            parameters: Mapping[str, ParameterValue],
        ) -> dict[str, ParameterValue]:
            return _conjugate_complex_gradient(gradient(parameters))

        return conjugated_gradient

    def raise_gradient_not_implemented(
        parameters: Mapping[str, ParameterValue],
    ) -> dict[str, ParameterValue]:
        msg = f"Gradient not implemented for back-end {backend}."
        raise NotImplementedError(msg)

    return raise_gradient_not_implemented


def _jit_estimator_core(core: Callable, backend: str) -> Callable:
    if backend == "jax":
        jax = _import_jax()
        return jax.jit(core)
    return core


def _convert_arrays_to_backend(data: DataSample, backend: str) -> DataSample:
    # move data arrays to the device once, so that JIT-compiled estimator calls
    # do not pay a host-to-device transfer on every evaluation
    if backend == "jax":
        jax = _import_jax()
        return {key: jax.numpy.asarray(array) for key, array in data.items()}
    return data


def _create_core_gradient(core: Callable, backend: str) -> Callable:
    """Create a JIT-compiled gradient of an estimator core, w.r.t. its parameters."""
    if backend == "jax":
        jax = _import_jax()
        raw_gradient = jax.jit(jax.grad(core, argnums=0))

        def gradient(
            parameters: Mapping[str, ParameterValue],
            *data_args: DataSample | Array | None,
        ) -> dict[str, ParameterValue]:
            return _conjugate_complex_gradient(raw_gradient(parameters, *data_args))

        return gradient

    def raise_gradient_not_implemented(
        parameters: Mapping[str, ParameterValue],
        *data_args: DataSample | Array | None,
    ) -> dict[str, ParameterValue]:
        msg = f"Gradient not implemented for back-end {backend}."
        raise NotImplementedError(msg)

    return raise_gradient_not_implemented


class ChiSquared(Estimator):
    r"""Chi-squared test estimator.

    .. math:: \chi^2 = \sum_{i=1}^n w_i\left(y_i - f_\mathbf{p}(x_i)\right)^2

    Args:
        function: A `.ParametrizedFunction` :math:`f_\mathbf{p}` with
            a set of free `~.ParametrizedFunction.parameters` :math:`\mathbf{p}`.
        domain: Input data-set :math:`\mathbf{x}` of :math:`n` events
            :math:`x_i` over which to compute :code:`function` :math:`f_\mathbf{p}`.
        observed_values: Observed values :math:`y_i`.
        weights: Optional weights :math:`w_i`. Default: :math:`w_i=1`
            (unweighted). A common choice is :math:`w_i = 1/\sigma_i^2`, with
            :math:`\sigma_i` the uncertainty in each measured value of :math:`y_i`.
        backend: Computational backend with which to compute the sum
            :math:`\sum_{i=1}^n`. By default, this is the backend of the
            :code:`function`, if it exposes one (see `.BackendFunction`).

    On the JAX backend, the full estimator and its analytic :meth:`gradient` are
    JIT-compiled once and cached over all further evaluations.

    .. seealso:: :doc:`/usage/chi-squared`
    """

    def __init__(
        self,
        function: ParametrizedFunction[DataSample, FloatArray],
        domain: DataSample,
        observed_values: Array,
        weights: Array | None = None,
        backend: str | None = None,
    ) -> None:
        backend = _determine_backend(function, backend)
        self.__domain = _convert_arrays_to_backend(domain, backend)
        if weights is None:
            ones = find_function("ones", backend)
            weights = ones(len(observed_values))
        converted = _convert_arrays_to_backend(
            {"observed_values": observed_values, "weights": weights}, backend
        )
        self.__observed_values = converted["observed_values"]
        self.__weights = converted["weights"]
        sum_function = find_function("sum", backend)

        def estimator(
            parameters: Mapping[str, ParameterType],
            domain: DataSample,
            observed_values: Array,
            weights: Array,
        ) -> float:
            computed_values = function(domain, parameters)
            chi_squared = weights * (computed_values - observed_values) ** 2
            return sum_function(chi_squared, axis=-1)

        self.__estimator = _jit_estimator_core(estimator, backend)
        self.__gradient = _create_core_gradient(estimator, backend)

    @overload
    def __call__(self, parameters: Mapping[str, ParameterValue]) -> float: ...
    @overload
    def __call__(self, parameters: Mapping[str, Array]) -> FloatArray: ...
    @overload
    def __call__(
        self, parameters: Mapping[str, ParameterType]
    ) -> float | FloatArray: ...
    def __call__(self, parameters: Mapping[str, ParameterType]) -> float | FloatArray:
        return self.__estimator(*self.__estimator_args(parameters))

    def gradient(
        self, parameters: Mapping[str, ParameterValue]
    ) -> dict[str, ParameterValue]:
        return self.__gradient(*self.__estimator_args(parameters))

    def __estimator_args(self, parameters: Mapping[str, ParameterType]) -> tuple:
        return (
            _coerce_parameter_types(parameters),
            self.__domain,
            self.__observed_values,
            self.__weights,
        )


class UnbinnedNLL(Estimator):
    r"""Unbinned negative log likelihood estimator.

    The **log likelihood** :math:`\log\mathcal{L}` for a given function
    :math:`f_\mathbf{p}: X^m \rightarrow \mathbb{R}` over :math:`N` data points
    :math:`\mathbf{x}` and over a (phase space) domain of :math:`n_\mathrm{phsp}` points
    :math:`\mathbf{x}_\mathrm{phsp}`, is given by:

    .. math::

        -\log\mathcal{L} = N\log\lambda -\sum_{i=1}^N \log\left(f_\mathbf{p}(x_i)\right)

    with :math:`\lambda` the normalization integral over :math:`f_\mathbf{p}`. The
    integral is computed numerically by averaging over a significantly large (phase
    space) domain sample :math:`\mathbf{x}_\mathrm{phsp}` of size :math:`n`:

    .. math::
        \lambda = \frac{\sum_{j=1}^n V f_\mathbf{p}(x_{\mathrm{phsp},j})}{n}.

    Args:
        function: A `.ParametrizedFunction` :math:`f_\mathbf{p}` that describes
            a distribution over a certain domain.
        data: The `.DataSample` :math:`\mathbf{x}` over which to compute
            :math:`f_\mathbf{p}`.
        phsp: The domain (phase space) with which the likelihood is normalized.
            When correcting for the detector efficiency, use a phase space sample that
            passed the detector reconstruction.
        phsp_volume: Optional phase space volume :math:`V`, used in the
            normalization factor. Default: :math:`V=1`.
        backend: The computational back-end with which the sums and averages
            should be computed. By default, this is the backend of the
            :code:`function`, if it exposes one (see `.BackendFunction`).

    On the JAX backend, the full estimator and its analytic :meth:`gradient` are
    JIT-compiled once and cached over all further evaluations.

    .. seealso:: :doc:`/usage/unbinned-fit`
    """

    def __init__(
        self,
        function: ParametrizedFunction[DataSample, FloatArray],
        data: DataSample,
        phsp: DataSample,
        phsp_volume: float = 1.0,
        backend: str | None = None,
    ) -> None:
        backend = _determine_backend(function, backend)
        self.__data = _convert_arrays_to_backend(dict(data), backend)
        converted_phsp = _convert_arrays_to_backend(dict(phsp), backend)
        self.__phsp = {k: v for k, v in converted_phsp.items() if k != "weights"}
        self.__phsp_weights = converted_phsp.get("weights")
        mean_function = find_function("mean", backend)
        sum_function = find_function("sum", backend)
        log_function = find_function("log", backend)

        def estimator(
            parameters: Mapping[str, ParameterType],
            data: DataSample,
            phsp: DataSample,
            phsp_weights: Array | None,
        ) -> float:
            bare_intensities = function(data, parameters)
            phsp_intensities = function(phsp, parameters)
            if phsp_weights is not None:
                phsp_intensities *= phsp_weights
            normalization_integral = phsp_volume * mean_function(
                phsp_intensities, axis=-1
            )
            log_normalization = bare_intensities.shape[-1] * log_function(
                normalization_integral
            )
            return log_normalization - sum_function(
                log_function(bare_intensities), axis=-1
            )

        self.__estimator = _jit_estimator_core(estimator, backend)
        self.__gradient = _create_core_gradient(estimator, backend)

    @overload
    def __call__(self, parameters: Mapping[str, ParameterValue]) -> float: ...
    @overload
    def __call__(self, parameters: Mapping[str, Array]) -> FloatArray: ...
    @overload
    def __call__(
        self, parameters: Mapping[str, ParameterType]
    ) -> float | FloatArray: ...
    def __call__(self, parameters: Mapping[str, ParameterType]) -> float | FloatArray:
        return self.__estimator(*self.__estimator_args(parameters))

    def gradient(
        self, parameters: Mapping[str, ParameterValue]
    ) -> dict[str, ParameterValue]:
        return self.__gradient(*self.__estimator_args(parameters))

    def __estimator_args(self, parameters: Mapping[str, ParameterType]) -> tuple:
        return (
            _coerce_parameter_types(parameters),
            self.__data,
            self.__phsp,
            self.__phsp_weights,
        )
