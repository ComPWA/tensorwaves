# cspell:ignore dphi, epem
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
    from expertsystem.io import load_pdg, write
    from expertsystem.particle import Parity, Particle
    from expertsystem.reaction import generate

    epem = Particle(
        name="EpEm",
        pid=12345678,
        mass=4.36,
        spin=1.0,
        parity=Parity(-1),
        c_parity=Parity(-1),
    )
    particles = load_pdg()
    particles.add(epem)

    result = generate(
        initial_state=[("EpEm", [-1])],
        final_state=[("D0", [0]), ("D-", [0]), ("pi+", [0])],
        allowed_intermediate_particles=["D(2)*(2460)+"],
        allowed_interaction_types="em",
        particles=particles,
    )
    print(list(x.name for x in result.get_intermediate_particles()))

    generator = HelicityAmplitudeGenerator()
    amplitude_model = generator.generate(result)
    amplitude_model.dynamics.set_non_dynamic("D(2)*(2460)+")
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
        WignerD(1, -1, 1, -phi1, theta1, phi1)
        * WignerD(2, 1, 0, -phi2, theta2, phi2)
        - 1
        * WignerD(1, -1, -1, -phi1, theta1, phi1)
        * WignerD(2, -1, 0, -phi2, theta2, phi2)
    ).doit()

    intensity = sympy.simplify(
        (amp * sympy.conjugate(amp)).expand(complex=True)
    )
    intensity = sympy.simplify(intensity.replace(phi2, dphi + phi1))

    assert sympy.im(intensity) == 0
    print(sympy.re(intensity))
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
        events=50000,
    )


@pytest.mark.parametrize(
    "angular_variable, expected_distribution_function",  # type: ignore
    [
        (  # x = cos(theta) distribution from epem decay
            "theta_3+4_2",
            lambda x: 1 + x * x,
        ),
        (  # phi distribution of the epem decay
            "phi_3+4_2",
            lambda x: 1,
        ),
        (  # x = cos(theta') distribution from D2*
            "theta_3_4_vs_2",
            lambda x: 1 - (2 * x * x - 1) ** 2,
        ),
        (  # phi' distribution of the D2* decay
            "phi_3_4_vs_2",
            lambda phi: 2 + cos(2 * phi),
        ),
        # ( # 2d distribution of the D2* decay
        #   ['theta_3_4_vs_2', 'phi_3_4_vs_2'],
        #   lambda x, phi: (1 - x**2) * (x**2) * (2 + cos(2 * phi)),
        # )
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
        make_plots=True,
    )
