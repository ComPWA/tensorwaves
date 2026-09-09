from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
import sympy as sp

from tensorwaves.data import NumpyDomainGenerator, NumpyUniformRNG
from tensorwaves.data.transform import SympyDataTransformer
from tensorwaves.estimator import ChiSquared, UnbinnedNLL, create_cached_function
from tensorwaves.function import ParametrizedBackendFunction, PositionalArgumentFunction
from tensorwaves.function.sympy import create_parametrized_function
from tensorwaves.optimizer.minuit import Minuit2

if TYPE_CHECKING:
    from tensorwaves.interface import DataSample, ParameterValue

NUMPY_RNG = np.random.default_rng(12345)


def gaussian(mu_: float, sigma_: float) -> ParametrizedBackendFunction:
    x, mu, sigma = sp.symbols("x, mu, sigma")
    return create_parametrized_function(
        expression=sp.exp(-(((x - mu) / sigma) ** 2) / 2),
        parameters={
            mu: mu_,
            sigma: sigma_,
        },
        backend="numpy",
    )


def gaussian_sum(
    a_1: float,
    mu_1: float,
    sigma_1: float,
    a_2: float,
    mu_2: float,
    sigma_2: float,
) -> ParametrizedBackendFunction:
    x, a1, mu1, sigma1, a2, mu2, sigma2 = sp.symbols(
        "x, a1, mu1, sigma1, a2, mu2, sigma2"
    )
    gaussian1 = (
        a1
        / (sigma1 * sp.sqrt(2.0 * math.pi))
        * sp.exp(-(((x - mu1) / sigma1) ** 2) / 2)
    )
    gaussian2 = (
        a2
        / (sigma2 * sp.sqrt(2.0 * math.pi))
        * sp.exp(-(((x - mu2) / sigma2) ** 2) / 2)
    )

    return create_parametrized_function(
        expression=gaussian1 + gaussian2,
        parameters={
            a1: a_1,
            mu1: mu_1,
            sigma1: sigma_1,
            a2: a_2,
            mu2: mu_2,
            sigma2: sigma_2,
        },
        backend="numpy",
    )


@pytest.fixture(scope="module")
def phsp() -> DataSample:
    rng = np.random.default_rng(12345)
    return {
        "x": rng.uniform(low=-2.0, high=5.0, size=10000),
    }


def describe_ChiSquared():
    @pytest.fixture
    def linear_function() -> ParametrizedBackendFunction:
        return ParametrizedBackendFunction(
            function=lambda a, b, x: a + b * x,
            argument_order=("a", "b", "x"),
            parameters={"a": 0, "b": 1},
        )

    x_data = {"x": np.array([0, 1, 2])}
    y_data = np.array([0, 1, 2])

    @pytest.mark.parametrize("backend", ["jax", "numpy", "tensorflow"])
    def it_sums_the_squared_residuals(backend: str, linear_function):
        estimator = ChiSquared(linear_function, x_data, y_data, backend=backend)
        assert estimator({}) == 0
        assert estimator({"b": 2}) == 5.0
        assert estimator({"a": 1, "b": 2}) == 14.0

    @pytest.mark.parametrize("backend", ["jax", "numpy", "tensorflow"])
    def it_scales_the_residuals_with_the_weights(backend: str, linear_function):
        estimator = ChiSquared(
            linear_function,
            x_data,
            y_data,
            weights=1 / (2 * np.ones(3)),
            backend=backend,
        )
        assert estimator({"a": 0, "b": 2}) == 2.5


def describe_create_cached_function():
    @pytest.fixture(params=["jax", "numba", "numpy", "tf"], scope="module")
    def backend(request) -> str:
        return request.param

    @pytest.fixture(scope="module")
    def expression_and_parameters() -> tuple[sp.Expr, dict[sp.Basic, int | float]]:
        a, b, c, d, x, y = sp.symbols("a b c d x y")
        expression = a * x + b * (c * x + d * y**2)
        return expression, {a: -2.5, b: 1.4, c: 0.8, d: 3.7}

    @pytest.fixture(scope="module")
    def cached_function_and_transformer(expression_and_parameters, backend: str):
        expression, parameter_defaults = expression_and_parameters
        return create_cached_function(
            expression,
            parameter_defaults,
            backend,
            free_parameters={sp.Symbol("a"), sp.Symbol("c")},
        )

    def it_returns_a_function_and_a_transformer(cached_function_and_transformer):
        cached_function, cache_transformer = cached_function_and_transformer
        assert isinstance(cached_function, ParametrizedBackendFunction)
        assert isinstance(cache_transformer, SympyDataTransformer)

    def it_puts_data_arguments_before_the_free_parameters(
        cached_function_and_transformer,
    ):
        cached_function, cache_transformer = cached_function_and_transformer
        assert cached_function.argument_order == ("f0", "x", "a", "c")
        assert set(cached_function.parameters) == {"a", "c"}
        assert set(cache_transformer.functions) == {"f0", "x"}

    def it_creates_transformer_functions_over_the_domain_variables(
        cached_function_and_transformer, expression_and_parameters
    ):
        _, cache_transformer = cached_function_and_transformer
        expression, parameter_defaults = expression_and_parameters
        domain_variables = expression.free_symbols - set(parameter_defaults)
        for func in cache_transformer.functions.values():
            assert isinstance(func, PositionalArgumentFunction)
            assert set(func.argument_order) == set(map(str, domain_variables))

    def it_reproduces_the_intensities_of_the_uncached_function(
        cached_function_and_transformer, expression_and_parameters, backend: str
    ):
        cached_function, cache_transformer = cached_function_and_transformer
        expression, parameter_defaults = expression_and_parameters
        function = create_parametrized_function(expression, parameter_defaults, backend)

        domain_generator = NumpyDomainGenerator({"x": (-1, +1), "y": (-1, +1)})
        domain = domain_generator.generate(100, NumpyUniformRNG())
        cached_domain = cache_transformer(domain)

        intensities = function(domain)
        cached_intensities = cached_function(cached_domain)
        np.testing.assert_allclose(intensities, cached_intensities)


def describe_UnbinnedNLL():
    @pytest.mark.parametrize(
        ("function", "data", "true_params"),
        [
            (
                gaussian(1.0, 0.1),
                {
                    "x": NUMPY_RNG.normal(0.5, 0.1, 1000),
                },
                {"mu": 0.5},
            ),
            (
                gaussian(1.0, 0.1),
                {
                    "x": NUMPY_RNG.normal(0.5, 0.3, 1000),
                },
                {"mu": 0.5, "sigma": 0.3},
            ),
            (
                gaussian_sum(1.0, 1.0, 0.1, 2.0, 2.0, 0.3),
                {
                    "x": np.append(
                        NUMPY_RNG.normal(
                            1.0,
                            0.1,
                            2000,
                        ),
                        NUMPY_RNG.normal(
                            2.0,
                            0.3,
                            1000,
                        ),
                    )
                },
                {"a2": 0.5},  # ratio should be A1/A2 = 2000/1000 -- A1=1 --> A2=0.5
            ),
            (
                gaussian_sum(1.0, 1.0, 0.1, 1.0, 2.0, 0.3),
                {
                    "x": np.append(
                        NUMPY_RNG.normal(
                            0.9,
                            0.3,
                            1000,
                        ),
                        NUMPY_RNG.normal(
                            2.5,
                            0.1,
                            1000,
                        ),
                    )
                },
                {"mu1": 0.9, "sigma1": 0.3, "mu2": 2.5, "sigma2": 0.1},
            ),
            (
                gaussian_sum(1.0, 1.0, 0.1, 2.0, 2.5, 0.3),
                {
                    "x": np.append(
                        NUMPY_RNG.normal(
                            0.9,
                            0.3,
                            2000,
                        ),
                        NUMPY_RNG.normal(
                            2.5,
                            0.1,
                            1000,
                        ),
                    )
                },
                {"mu1": 0.9, "sigma1": 0.3, "a2": 0.5, "sigma2": 0.1},
            ),
        ],
    )
    def it_recovers_the_parameters_that_generated_the_data(
        function,
        data: DataSample,
        true_params: dict[str, ParameterValue],
        phsp: DataSample,
    ):
        estimator = UnbinnedNLL(
            function,
            data,
            phsp,
            phsp_volume=6.0,
        )
        minuit2 = Minuit2()
        fit_result = minuit2.optimize(
            estimator,
            initial_parameters=true_params,
        )

        par_values = fit_result.parameter_values
        par_errors = fit_result.parameter_errors
        assert par_errors is not None

        assert set(par_values) == set(true_params)
        for par_name, par_value in true_params.items():
            par_error = par_errors[par_name]
            assert isinstance(par_error, float)
            assert abs(par_values[par_name] - par_value) < 4.0 * par_error
            assert par_value == pytest.approx(par_values[par_name], rel=0.1)
