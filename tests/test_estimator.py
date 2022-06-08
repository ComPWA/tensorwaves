# pylint: disable=invalid-name import-error redefined-outer-name
# pylint: disable=invalid-name too-many-locals unsubscriptable-object
from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from tensorwaves.data import NumpyDomainGenerator, NumpyUniformRNG
from tensorwaves.data.transform import SympyDataTransformer
from tensorwaves.estimator import (
    ChiSquared,
    UnbinnedNLL,
    create_cached_function,
)
from tensorwaves.function import (
    ParametrizedBackendFunction,
    PositionalArgumentFunction,
)
from tensorwaves.function.sympy import create_parametrized_function
from tensorwaves.interface import DataSample, ParameterValue
from tensorwaves.optimizer.minuit import Minuit2


class TestChiSquared:
    @pytest.mark.parametrize("backend", ["jax", "numpy", "tensorflow"])
    def test_call(self, backend):
        x_data = {"x": np.array([0, 1, 2])}
        y_data = np.array([0, 1, 2])
        function = ParametrizedBackendFunction(
            function=lambda a, b, x: a + b * x,
            argument_order=("a", "b", "x"),
            parameters={"a": 0, "b": 1},
        )
        estimator = ChiSquared(function, x_data, y_data, backend=backend)
        assert estimator({}) == 0
        assert estimator({"b": 2}) == 5.0
        assert estimator({"a": 1, "b": 2}) == 14.0
        estimator = ChiSquared(
            function,
            x_data,
            y_data,
            weights=1 / (2 * np.ones(3)),
            backend=backend,
        )
        assert estimator({"a": 0, "b": 2}) == 2.5


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


@pytest.mark.parametrize("backend", ["jax", "numba", "numpy", "tf"])
def test_create_cached_function(backend):
    __symbols: tuple[sp.Symbol, ...] = sp.symbols("a b c d x y")
    a, b, c, d, x, y = __symbols
    expression = a * x + b * (c * x + d * y**2)
    parameter_defaults = {a: -2.5, b: 1.4, c: 0.8, d: 3.7}

    function = create_parametrized_function(
        expression, parameter_defaults, backend
    )
    cached_function, cache_transformer = create_cached_function(
        expression, parameter_defaults, backend, free_parameters={a, c}
    )

    assert isinstance(cached_function, ParametrizedBackendFunction)
    assert isinstance(cache_transformer, SympyDataTransformer)
    assert cached_function.argument_order == ("a", "c", "f0", "x")
    assert set(cached_function.parameters) == {"a", "c"}
    assert set(cache_transformer.functions) == {"f0", "x"}

    domain_variables = expression.free_symbols - set(parameter_defaults)
    for func in cache_transformer.functions.values():
        assert isinstance(func, PositionalArgumentFunction)
        assert set(func.argument_order) == set(map(str, domain_variables))

    domain_generator = NumpyDomainGenerator({"x": (-1, +1), "y": (-1, +1)})
    rng = NumpyUniformRNG()
    domain = domain_generator.generate(100, rng)
    cached_domain = cache_transformer(domain)

    intensities = function(domain)
    cached_intensities = cached_function(cached_domain)
    np.testing.assert_allclose(intensities, cached_intensities)


NUMPY_RNG = np.random.default_rng(12345)


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
            {
                "a2": 0.5
            },  # ratio should be A1/A2 = 2000/1000 -- A1=1 --> A2=0.5
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
def test_sympy_unbinned_nll(
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
