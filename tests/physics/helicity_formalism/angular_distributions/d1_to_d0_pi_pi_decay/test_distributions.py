# cspell:ignore dphi
# pylint: disable=import-outside-toplevel,redefined-outer-name
import os
from math import cos
from typing import Any, Callable, List, Tuple

import numpy as np
import pytest


# Use this function to reproduce the model file.
# Note the normalization part has been removed!
def generate_model() -> None:
    from expertsystem.amplitude.helicity_decay import (
        HelicityAmplitudeGenerator,
    )
    from expertsystem.io import write
    from expertsystem.reaction import generate

    result = generate(
        initial_state=[("D(1)(2420)0", [-1])],
        final_state=[("D0", [0]), ("pi-", [0]), ("pi+", [0])],
        allowed_intermediate_particles=["D*"],
        allowed_interaction_types="strong",
    )
    print(list(x.name for x in result.get_intermediate_particles()))

    generator = HelicityAmplitudeGenerator()
    amplitude_model = generator.generate(result)
    amplitude_model.dynamics.set_non_dynamic("D*(2010)+")
    write(amplitude_model, "model.yml")


# Use this function to reproduce the theoretical predictions.
def calc_distributions() -> List[Tuple[str, Any]]:
    import sympy
    from sympy.abc import symbols
    from sympy.physics.quantum.spin import WignerD

    theta1, phi1, theta2, phi2, dphi = symbols(
        "theta1,phi1,theta2,phi2,dphi", real=True
    )

    amp = (
        WignerD(1, -1, -1, -phi1, theta1, 0)
        * WignerD(1, -1, 0, -phi2, theta2, 0)
        + 0.5
        * WignerD(1, -1, 0, -phi1, theta1, 0)
        * WignerD(1, 0, 0, -phi2, theta2, 0)
        + WignerD(1, -1, 1, -phi1, theta1, 0)
        * WignerD(1, 1, 0, -phi2, theta2, 0)
    ).doit()

    intensity = sympy.simplify(
        (amp * sympy.conjugate(amp)).expand(complex=True)
    )
    intensity = intensity.replace(phi2, dphi + phi1)

    assert sympy.im(intensity) == 0
    return [
        (
            "theta1 dependency:",
            sympy.trigsimp(
                sympy.re(
                    sympy.integrate(
                        intensity * sympy.sin(theta2),  # jacobi determinant!
                        (phi1, -sympy.pi, sympy.pi),
                        (theta2, 0, sympy.pi),
                        (dphi, -sympy.pi, sympy.pi),
                    )
                )
            ).doit(),
        ),
        (
            "theta2 dependency:",
            sympy.trigsimp(
                sympy.re(
                    sympy.integrate(
                        intensity * sympy.sin(theta1),  # jacobi determinant!
                        (phi1, -sympy.pi, sympy.pi),
                        (theta1, 0.0, sympy.pi),
                        (dphi, -sympy.pi, sympy.pi),
                    )
                )
            ).doit(),
        ),
        (
            "phi1 dependency:",
            sympy.trigsimp(
                sympy.re(
                    sympy.integrate(
                        intensity
                        * sympy.sin(theta1)  # jacobi determinant!
                        * sympy.sin(theta2),  # jacobi determinant!
                        (dphi, -sympy.pi, sympy.pi),
                        (theta1, 0.0, sympy.pi),
                        (theta2, 0.0, sympy.pi),
                    )
                )
            ).doit(),
        ),
        (
            "dphi dependency:",
            sympy.trigsimp(
                sympy.re(
                    sympy.integrate(
                        intensity
                        * sympy.sin(theta1)  # jacobi determinant!
                        * sympy.sin(theta2),  # jacobi determinant!
                        (phi1, -sympy.pi, sympy.pi),
                        (theta1, 0.0, sympy.pi),
                        (theta2, 0.0, sympy.pi),
                    )
                )
            ).doit(),
        ),
    ]


@pytest.fixture(scope="module")
def intensity_dataset(
    generate_dataset: Callable,
) -> np.ndarray:
    thisdirectory = os.path.dirname(os.path.realpath(__file__))
    return generate_dataset(
        model_filename=thisdirectory + "/model.yml",
        events=30000,
    )


@pytest.mark.parametrize(
    "angular_variable, expected_distribution_function",  # type: ignore
    [
        (  # x = cos(theta) distribution from D1 decay
            "theta_3+4_2",
            lambda x: 1.25 + 0.75 * x * x,
        ),
        (  # x = cos(theta') distribution from D*
            "theta_3_4_vs_2",
            lambda x: 1 - 0.75 * x * x,
        ),
        (  # phi distribution of the D* decay
            "phi_3_4_vs_2",
            lambda x: 1 - 1 / 2.25 * cos(2 * x),
        ),
    ],  # type: ignore
)
def test_distributions_reduced_chi2(
    angular_variable: str,
    expected_distribution_function: Callable,
    intensity_dataset,
    test_angular_distribution,
    chisquare_test,
) -> None:

    test_angular_distribution(
        intensity_dataset,
        angular_variable,
        expected_distribution_function,
        chisquare_test,
        bins=180,
        make_plots=False,
    )


@pytest.mark.parametrize(
    "angular_variable, expected_distribution_function",  # type: ignore
    [
        (  # x = cos(theta) distribution from D1 decay
            "theta_3+4_2",
            lambda x: 1.25 + 0.75 * x * x,
        ),
        (  # x = cos(theta') distribution from D*
            "theta_3_4_vs_2",
            lambda x: 1 - 0.75 * x * x,
        ),
        (  # phi distribution of the D* decay
            "phi_3_4_vs_2",
            lambda x: 1 - 1 / 2.25 * cos(2 * x),
        ),
    ],  # type: ignore
)
def test_distributions_residuals(
    angular_variable: str,
    expected_distribution_function: Callable,
    intensity_dataset,
    test_angular_distribution,
    residual_test,
) -> None:

    test_angular_distribution(
        intensity_dataset,
        angular_variable,
        expected_distribution_function,
        residual_test,
        bins=180,
        make_plots=False,
    )
