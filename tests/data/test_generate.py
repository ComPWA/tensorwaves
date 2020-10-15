import pytest

from tensorwaves.data.generate import generate_phsp
from tensorwaves.physics.helicity_formalism.kinematics import (
    HelicityKinematics,
    ParticleReactionKinematicsInfo,
)


@pytest.mark.parametrize(
    "sample_size, initial_state_names, final_state_names, expected_shape",
    [
        (
            5000,
            "J/psi(1S)",
            ("pi0", "pi0", "pi0"),
            (3, 5000, 4),
        ),
        (
            320,
            ("J/psi(1S)"),
            ("pi0", "pi0", "pi0", "gamma"),
            (4, 320, 4),
        ),
        (
            250,
            "J/psi(1S)",
            ("pi0", "pi0", "pi0", "pi0", "gamma"),
            (5, 250, 4),
        ),
    ],
)
def test_shape_generate_phsp(
    sample_size, initial_state_names, final_state_names, expected_shape, pdg
):
    reaction_info = ParticleReactionKinematicsInfo(
        initial_state_names=initial_state_names,
        final_state_names=final_state_names,
        particles=pdg,
    )
    kin = HelicityKinematics(reaction_info)
    sample = generate_phsp(sample_size, kin)
    assert sample.shape == expected_shape
