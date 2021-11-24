# pylint: disable=invalid-name import-error redefined-outer-name unsubscriptable-object
import math
from typing import Dict

import numpy as np
import pytest
import sympy as sp

from tensorwaves.estimator import UnbinnedNLL
from tensorwaves.function import ParametrizedBackendFunction
from tensorwaves.function.sympy import create_function
from tensorwaves.interface import DataSample, ParameterValue
from tensorwaves.optimizer.minuit import Minuit2


def gaussian(mu_: float, sigma_: float) -> ParametrizedBackendFunction:
    x, mu, sigma = sp.symbols("x, mu, sigma")
    return create_function(
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

    return create_function(
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
def phsp_dataset() -> DataSample:
    rng = np.random.default_rng(12345)
    return {
        "x": rng.uniform(low=-2.0, high=5.0, size=10000),
    }


NUMPY_RNG = np.random.default_rng(12345)


@pytest.mark.parametrize(
    ("function", "dataset", "true_params"),
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
    dataset: DataSample,
    true_params: Dict[str, ParameterValue],
    phsp_dataset: DataSample,
):
    estimator = UnbinnedNLL(
        function,
        dataset,
        phsp_dataset,
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
