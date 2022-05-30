# pylint: disable=invalid-name, redefined-outer-name
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from tensorwaves.estimator import UnbinnedNLL
from tensorwaves.function.sympy import create_parametrized_function
from tensorwaves.interface import DataSample, Function
from tensorwaves.optimizer import Minuit2, ScipyMinimizer


def gaussian(x: sp.Symbol, mu: sp.Symbol, sigma: sp.Symbol) -> sp.Expr:
    return sp.exp(-(((x - mu) / sigma) ** 2) / 2)


def poisson(x: sp.Symbol, k) -> sp.Expr:
    return x**k * sp.exp(-x) / sp.factorial(k)


symbols = sp.symbols("x y (a:c) mu_(:2) sigma_(:2) omega")
x, y, a, b, c, mu1, mu2, sigma1, sigma2, omega = symbols
expression = (
    a * gaussian(x, mu1, sigma1)
    + b * gaussian(x, mu2, sigma2)
    + c * poisson(x, k=2)
) * sp.cos(y * omega) ** 2

domain_boundaries = {"x": (0, 5), "y": (-np.pi, +np.pi)}
parameter_defaults = {
    a: 0.15,
    b: 0.05,
    c: 0.3,
    mu1: 1.0,
    mu2: 2.7,
    omega: 0.5,
    sigma1: 0.3,
    sigma2: 0.5,
}
initial_parameters = {
    "a": 0.2,
    "b": 0.1,
    "c": 0.2,
    "mu_0": 0.9,
    "sigma_0": 0.4,
    "sigma_1": 0.4,
}


def _generate_domain(
    size: int,
    rng: np.random.Generator,
) -> DataSample:
    return {
        var_name: rng.uniform(size=size, low=low, high=high)
        for var_name, (low, high) in domain_boundaries.items()
    }


def _generate_data(
    size: int,
    function: Function,
    rng: np.random.Generator,
    bunch_size: int = 10_000,
) -> DataSample:
    # pylint: disable=line-too-long
    collected_sample = {var: np.array([]) for var in domain_boundaries}  # type: ignore[var-annotated]
    some_variable = next(iter(domain_boundaries))
    while len(collected_sample[some_variable]) < size:
        phsp = _generate_domain(bunch_size, rng)
        y_values = function(phsp)
        y_max = np.max(y_values)
        random_y_values = rng.uniform(size=bunch_size, high=y_max)
        hit_and_miss_sample = {
            var: phsp[var][random_y_values < y_values]
            for var in domain_boundaries
        }
        collected_sample = {
            var: np.concatenate(
                [collected_sample[var], hit_and_miss_sample[var]]
            )
            for var in domain_boundaries
        }
    return {var: collected_sample[var][:size] for var in domain_boundaries}


def generate_data_and_domain(
    backend: str, n_domain: int, n_data: int
) -> tuple[DataSample, DataSample]:
    function = create_parametrized_function(
        expression=expression,
        parameters=parameter_defaults,
        backend=backend,
    )
    rng = np.random.default_rng(seed=0)
    domain = _generate_domain(n_domain, rng)
    data = _generate_data(n_data, function, rng)
    return domain, data


@pytest.mark.benchmark(group="data-simple")
@pytest.mark.parametrize("backend", ["jax", "numpy", "numba", "tf"])
@pytest.mark.parametrize("size", [3_000])
def test_data(backend, benchmark, size):
    domain, data = benchmark(
        generate_data_and_domain, backend, n_data=size, n_domain=10 * size
    )
    assert pytest.approx(domain["x"][0]) == 3.18481
    assert pytest.approx(data["x"][0]) == 3.46731


@pytest.mark.benchmark(group="fit-simple")
@pytest.mark.parametrize("backend", ["jax", "numpy", "numba", "tf"])
@pytest.mark.parametrize("optimizer_type", [Minuit2, ScipyMinimizer])
@pytest.mark.parametrize("size", [1_000])
def test_fit(  # pylint: disable=too-many-locals
    backend: str,
    benchmark,
    optimizer_type: (type[Minuit2] | type[ScipyMinimizer]),
    size: int,
):
    domain, data = generate_data_and_domain(
        backend, n_data=size, n_domain=10 * size
    )
    function = create_parametrized_function(
        expression=expression,
        parameters=parameter_defaults,
        backend=backend,
    )

    original_parameters = function.parameters
    estimator = UnbinnedNLL(function, data, domain, backend=backend)  # type: ignore[arg-type]
    original_nll = estimator(function.parameters)
    optimizer = optimizer_type()
    result = benchmark(optimizer.optimize, estimator, function.parameters)

    assert pytest.approx(result.estimator_value, rel=1e-2) == original_nll
    if optimizer_type not in {ScipyMinimizer}:
        assert result.minimum_valid
    for par in function.parameters:
        original_value = original_parameters[par]
        converged_value = result.parameter_values[par]
        assert pytest.approx(original_value, rel=0.3) == converged_value
