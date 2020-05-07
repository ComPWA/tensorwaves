"""Tests the generate facade functions."""

import pytest

from tensorwaves.data.generate import generate_phsp
from tensorwaves.physics.helicityformalism.kinematics import (
    HelicityKinematics,
    ParticleReactionKinematicsInfo,
)

__PARTICLES = {
    "J/psi": {"Mass": {"Value": 3.096900}},
    "gamma": {"Mass": {"Value": 0.0}},
    "pi0": {"Mass": {"Value": 0.1349766}},
}


@pytest.mark.parametrize(
    "sample_size, kinematics_info, expected_shape",
    [
        (
            5000,
            {
                "initial_state_names": "J/psi",
                "final_state_names": ("pi0", "pi0", "pi0"),
                "particle_dict": __PARTICLES,
            },
            (3, 5000, 4),
        ),
        (
            320,
            {
                "initial_state_names": ("J/psi"),
                "final_state_names": ("pi0", "pi0", "pi0", "gamma"),
                "particle_dict": __PARTICLES,
            },
            (4, 320, 4),
        ),
        (
            250,
            {
                "initial_state_names": "J/psi",
                "final_state_names": ("pi0", "pi0", "pi0", "pi0", "gamma"),
                "particle_dict": __PARTICLES,
            },
            (5, 250, 4),
        ),
    ],
)
def test_shape_generate_phsp(sample_size, kinematics_info, expected_shape):
    reaction_info = ParticleReactionKinematicsInfo(**kinematics_info)
    kin = HelicityKinematics(reaction_info)
    sample = generate_phsp(sample_size, kin)
    assert sample.shape == expected_shape
