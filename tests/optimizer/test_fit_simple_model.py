# pylint: disable=invalid-name, redefined-outer-name
from __future__ import annotations

from pathlib import Path

import iminuit
import numpy as np
import pytest
import sympy as sp

from tensorwaves.estimator import UnbinnedNLL
from tensorwaves.function.sympy import create_parametrized_function
from tensorwaves.interface import DataSample, Function
from tensorwaves.optimizer import Minuit2, ScipyMinimizer
from tensorwaves.optimizer.callbacks import (
    CallbackList,
    CSVSummary,
    TFSummary,
    YAMLSummary,
)


def generate_domain(
    size: int,
    boundaries: dict[str, tuple[float, float]],
    rng: np.random.Generator,
) -> DataSample:
    return {
        var_name: rng.uniform(size=size, low=low, high=high)
        for var_name, (low, high) in boundaries.items()
    }


def generate_data(
    size: int,
    boundaries: dict[str, tuple[float, float]],
    function: Function,
    rng: np.random.Generator,
    bunch_size: int = 10_000,
) -> DataSample:
    collected_sample = {var: np.array([]) for var in boundaries}  # type: ignore[var-annotated]
    some_variable = next(iter(boundaries))
    while len(collected_sample[some_variable]) < size:
        phsp = generate_domain(bunch_size, boundaries, rng)
        y_values = function(phsp)
        y_max = np.max(y_values)
        random_y_values = rng.uniform(size=bunch_size, high=y_max)
        hit_and_miss_sample = {
            var: phsp[var][random_y_values < y_values] for var in boundaries
        }
        collected_sample = {
            var: np.concatenate(
                [collected_sample[var], hit_and_miss_sample[var]]
            )
            for var in boundaries
        }
    return {var: collected_sample[var][:size] for var in boundaries}


def gaussian(x: sp.Symbol, mu: sp.Symbol, sigma: sp.Symbol) -> sp.Expr:
    return sp.exp(-(((x - mu) / sigma) ** 2) / 2)


def poisson(x: sp.Symbol, k) -> sp.Expr:
    return x**k * sp.exp(-x) / sp.factorial(k)


@pytest.fixture(scope="session")
def expression_and_parameters() -> tuple[sp.Expr, dict[sp.Symbol, float]]:
    symbols: tuple[sp.Symbol, ...] = sp.symbols(
        "x y (a:c) mu_(:2) sigma_(:2) omega"
    )
    x, y, a, b, c, mu1, mu2, sigma1, sigma2, omega = symbols
    expression = (
        a * gaussian(x, mu1, sigma1)
        + b * gaussian(x, mu2, sigma2)
        + c * poisson(x, k=2)
    ) * sp.cos(y * omega) ** 2
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
    assert set(symbols) - {x, y} == set(parameter_defaults)
    return expression, parameter_defaults


@pytest.fixture(scope="session")
def domain_and_data_sample(
    expression_and_parameters: tuple[sp.Expr, dict[sp.Symbol, float]]
) -> tuple[DataSample, DataSample]:
    expression, parameter_defaults = expression_and_parameters
    function = create_parametrized_function(
        expression=expression,
        parameters=parameter_defaults,
        backend="jax",
    )
    boundaries = {"x": (0, 5), "y": (-np.pi, +np.pi)}
    rng = np.random.default_rng(seed=0)
    domain = generate_domain(30_000, boundaries, rng)
    data = generate_data(3_000, boundaries, function, rng)
    # check if samples are deterministic
    assert pytest.approx(domain["x"][0]) == 3.18481
    assert pytest.approx(data["x"][0]) == 3.46731
    return domain, data


@pytest.mark.parametrize("optimizer_type", [Minuit2, ScipyMinimizer])
@pytest.mark.parametrize("backend", ["jax", "numpy", "numba", "tf"])
def test_optimize_all_parameters(  # pylint: disable=too-many-locals
    backend: str,
    domain_and_data_sample: tuple[DataSample, DataSample],
    expression_and_parameters: tuple[sp.Expr, dict[sp.Symbol, float]],
    optimizer_type: (type[Minuit2] | type[ScipyMinimizer]),
    output_dir: Path,
):
    domain, data = domain_and_data_sample
    expression, parameter_defaults = expression_and_parameters
    function = create_parametrized_function(
        expression=expression,
        parameters=parameter_defaults,
        backend=backend,
    )
    original_parameters = function.parameters
    estimator = UnbinnedNLL(function, data, domain, backend=backend)  # type: ignore[arg-type]
    original_nll = estimator(function.parameters)

    callback_file = (
        output_dir / f"simple_fit_{backend}_{optimizer_type.__name__}"
    )
    callbacks = [
        CSVSummary(f"{callback_file}.csv"),
        YAMLSummary(f"{callback_file}.yml"),
    ]
    try:
        # pylint: disable=import-outside-toplevel
        # pyright: reportUnusedImport=false
        import tensorflow  # noqa: F401

        callbacks.append(TFSummary())
    except ImportError:
        pass

    optimizer = optimizer_type(callback=CallbackList(callbacks))
    result = optimizer.optimize(estimator, function.parameters)

    csv = CSVSummary.load_latest_parameters(f"{callback_file}.csv")
    assert csv["function_call"] == result.function_calls
    assert csv["estimator_type"] == UnbinnedNLL.__name__
    assert pytest.approx(csv["estimator_value"]) == result.estimator_value
    for par in function.parameters:
        assert pytest.approx(csv[par]) == result.parameter_values[par]

    yaml = YAMLSummary.load_latest_parameters(f"{callback_file}.yml")
    for par in function.parameters:
        assert pytest.approx(yaml[par]) == result.parameter_values[par]

    assert pytest.approx(result.estimator_value, rel=5e-3) == original_nll
    if optimizer_type not in {ScipyMinimizer}:
        assert result.minimum_valid
    for par in function.parameters:
        original_value = original_parameters[par]
        converged_value = result.parameter_values[par]
        assert pytest.approx(original_value, rel=0.2) == converged_value


@pytest.mark.parametrize(
    ("tol", "expected_parameter_values"),
    [
        (
            0.1,  # iminuit default tolerance
            {
                "a": 0.15679884056468815,
                "b": 0.051281396032855225,
                "c": 0.26265501744837677,
                "mu_0": 0.9871104323476636,
                "mu_1": 2.6947038781339754,
                "omega": 0.4982768824682492,
                "sigma_0": 0.3075629925771585,
                "sigma_1": 0.5768191611084318,
            },
        ),
        (
            2.0,
            {
                "a": 0.15676480837709061,
                "b": 0.051242383278715484,
                "c": 0.26300711305648883,
                "mu_0": 0.9870594159578658,
                "mu_1": 2.694891339245927,
                "omega": 0.4982886866357587,
                "sigma_0": 0.30740850647117296,
                "sigma_1": 0.5760794793407019,
            },
        ),
    ],
)
def test_tweak_minuit(
    domain_and_data_sample: tuple[DataSample, DataSample],
    expression_and_parameters: tuple[sp.Expr, dict[sp.Symbol, float]],
    tol: float,
    expected_parameter_values: dict[str, float],
):
    domain, data = domain_and_data_sample
    expression, parameter_defaults = expression_and_parameters
    backend = "jax"
    function = create_parametrized_function(
        expression=expression,
        parameters=parameter_defaults,
        backend=backend,
    )

    estimator = UnbinnedNLL(function, data, domain, backend=backend)
    assert pytest.approx(estimator(function.parameters)) == -1460.287922492544

    def tweak_minuit(minuit: iminuit.Minuit) -> None:
        minuit.tol = tol

    optimizer = Minuit2(minuit_modifier=tweak_minuit)
    result = optimizer.optimize(estimator, function.parameters)

    assert pytest.approx(result.estimator_value) == -1463.062749889655
    assert pytest.approx(result.parameter_values) == expected_parameter_values
