# cspell:ignore dphi
# pylint: disable=import-outside-toplevel,redefined-outer-name
import os
from math import cos
from typing import Any, Callable, List, Optional, Tuple

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

    generator = HelicityAmplitudeGenerator()
    amplitude_model = generator.generate(result)
    amplitude_model.dynamics.set_non_dynamic("D*(2010)+")
    write(amplitude_model, "model.yml")


# Use this function to reproduce the theoretical predictions.
def calc_distributions() -> List[Tuple[str, Any]]:
    import sympy
    from sympy.abc import symbols
    from sympy.physics.quantum.spin import WignerD

    def calculate_integral(
        intensity: Any,
        integration_variables: List[Any],
        jacobi_determinant: Optional[Any] = None,
    ) -> Any:
        if jacobi_determinant is None:
            for int_var in integration_variables:
                if "theta" in int_var.name:
                    intensity *= sympy.sin(int_var)
        else:
            intensity *= jacobi_determinant
        return sympy.trigsimp(
            sympy.re(
                sympy.integrate(
                    intensity,
                    *(
                        (x, -sympy.pi, sympy.pi)
                        if "phi" in x.name
                        else (x, 0, sympy.pi)
                        for x in integration_variables
                    ),
                )
            )
        ).doit()

    theta1, phi1, theta2, phi2 = symbols("theta1,phi1,theta2,phi2", real=True)

    # The phi1 dependency vanishes completely, hence phi2 can be seen as the
    # difference between the two phi angles.
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

    assert sympy.im(intensity) == 0

    all_variables = [theta1, phi1, theta2, phi2]
    return [
        (
            f"{var.name} dependency:",
            calculate_integral(
                intensity,
                all_variables[0:i] + all_variables[i + 1 :],
            ),
        )
        for i, var in enumerate(all_variables)
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
        bins=120,
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
        bins=120,
        make_plots=False,
    )
