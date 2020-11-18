import pytest

from tensorwaves.data.generate import generate_phsp
from tensorwaves.physics.helicity_formalism.kinematics import (
    HelicityKinematics,
    ParticleReactionKinematicsInfo,
)


@pytest.mark.parametrize(
    "initial_state_names, final_state_names",
    [
        (
            "J/psi(1S)",
            ("pi0", "pi0", "pi0"),
        ),
        (
            ("J/psi(1S)"),
            ("pi0", "pi0", "pi0", "gamma"),
        ),
        (
            "J/psi(1S)",
            ("pi0", "pi0", "pi0", "pi0", "gamma"),
        ),
    ],
)
def test_shape_generate_phsp(initial_state_names, final_state_names, pdg):
    reaction_info = ParticleReactionKinematicsInfo(
        initial_state_names=initial_state_names,
        final_state_names=final_state_names,
        particles=pdg,
    )
    kin = HelicityKinematics(reaction_info)
    sample_size = 3
    sample = generate_phsp(sample_size, kin)
    assert sample.shape == (len(final_state_names), sample_size, 4)
