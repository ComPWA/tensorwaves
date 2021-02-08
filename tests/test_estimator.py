# pylint: disable=invalid-name, redefined-outer-name

import math

import numpy as np
import pytest
import sympy as sy

from tensorwaves.estimator import SympyUnbinnedNLL
from tensorwaves.optimizer.minuit import Minuit2
from tensorwaves.physics.amplitude import SympyModel


def gaussian(mu_, sigma_):
    x, mu, sigma = sy.symbols("x, mu, sigma")

    return SympyModel(
        expression=(sy.exp(-(((x - mu) / sigma) ** 2) / 2)),
        parameters={
            mu: mu_,
            sigma: sigma_,
        },
        variables={x: None},
    )


def gaussian_sum(a_1, mu_1, sigma_1, a_2, mu_2, sigma_2):
    x, a1, mu1, sigma1, a2, mu2, sigma2 = sy.symbols(
        "x, a1, mu1, sigma1, a2, mu2, sigma2"
    )
    gaussian1 = (
        a1
        / (sigma1 * sy.sqrt(2.0 * math.pi))
        * sy.exp(-(((x - mu1) / sigma1) ** 2) / 2)
    )
    gaussian2 = (
        a2
        / (sigma2 * sy.sqrt(2.0 * math.pi))
        * sy.exp(-(((x - mu2) / sigma2) ** 2) / 2)
    )

    return SympyModel(
        expression=gaussian1 + gaussian2,
        parameters={
            a1: a_1,
            mu1: mu_1,
            sigma1: sigma_1,
            a2: a_2,
            mu2: mu_2,
            sigma2: sigma_2,
        },
        variables={x: None},
    )


@pytest.fixture(scope="module")
def phsp_dataset():
    rng = np.random.default_rng(12345)
    return {"x": rng.uniform(low=-2.0, high=5.0, size=10000)}


__np_rng = np.random.default_rng(12345)


@pytest.mark.parametrize(
    "model, dataset, true_params",
    [
        (
            gaussian(1.0, 0.1),
            {
                "x": __np_rng.normal(0.5, 0.1, 1000),
            },
            {"mu": 0.5},
        ),
        (
            gaussian(1.0, 0.1),
            {
                "x": __np_rng.normal(0.5, 0.3, 1000),
            },
            {"mu": 0.5, "sigma": 0.3},
        ),
        (
            gaussian_sum(1.0, 1.0, 0.1, 2.0, 2.0, 0.3),
            {
                "x": np.append(
                    __np_rng.normal(
                        1.0,
                        0.1,
                        2000,
                    ),
                    __np_rng.normal(
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
                    __np_rng.normal(
                        0.9,
                        0.3,
                        1000,
                    ),
                    __np_rng.normal(
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
                    __np_rng.normal(
                        0.9,
                        0.3,
                        2000,
                    ),
                    __np_rng.normal(
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
    model: SympyModel, dataset: dict, true_params: dict, phsp_dataset: dict
):
    estimator = SympyUnbinnedNLL(model, dataset, phsp_dataset, phsp_volume=6.0)
    minuit2 = Minuit2()
    result = minuit2.optimize(
        estimator,
        initial_parameters={
            k.name: v
            for k, v in model.parameters.items()
            if k.name in true_params
        },
    )

    par_values = result["parameter_values"]
    par_errors = result["parameter_errors"]

    assert set(par_values) == set(true_params)
    for par_name, par_value in true_params.items():
        assert (
            abs(par_values[par_name] - par_value) < 4.0 * par_errors[par_name]
        )
        assert par_value == pytest.approx(par_values[par_name], rel=0.1)
