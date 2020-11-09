import expertsystem.amplitude.model as es
import pytest

from tensorwaves.data.generate import generate_data, generate_phsp
from tensorwaves.physics.helicity_formalism.amplitude import IntensityBuilder
from tensorwaves.physics.helicity_formalism.kinematics import (
    HelicityKinematics,
    ParticleReactionKinematicsInfo,
)


@pytest.mark.parametrize(
    "sample_size, initial_state_names, final_state_names",
    [
        (
            5000,
            "J/psi(1S)",
            ("pi0", "pi0", "pi0"),
        ),
        (
            320,
            ("J/psi(1S)"),
            ("pi0", "pi0", "pi0", "gamma"),
        ),
        (
            250,
            "J/psi(1S)",
            ("pi0", "pi0", "pi0", "pi0", "gamma"),
        ),
    ],
)
def test_generate_phsp(
    sample_size, initial_state_names, final_state_names, pdg
):
    reaction_info = ParticleReactionKinematicsInfo(
        initial_state_names=initial_state_names,
        final_state_names=final_state_names,
        particles=pdg,
    )
    kin = HelicityKinematics(reaction_info)
    sample = generate_phsp(sample_size, kin)
    assert len(sample) == sample_size


def test_generate_data(canonical_model: es.AmplitudeModel):
    n_phsp = 1000
    n_data = 100
    model = canonical_model
    kinematics = HelicityKinematics.from_model(model)
    phsp_sample = generate_phsp(n_phsp, kinematics)
    builder = IntensityBuilder(model.particles, kinematics, phsp_sample)
    intensity = builder.create_intensity(model)
    data_sample = generate_data(n_data, kinematics, intensity)
    assert len(data_sample) == len(model.kinematics.final_state)
    for sample in data_sample:
        assert len(sample) == n_data
    data_set = kinematics.convert(data_sample)
    assert set(data_set) == {
        "mSq_2",
        "mSq_2_3_4",
        "mSq_3",
        "mSq_3_4",
        "mSq_4",
        "phi+3+4_vs_2",
        "phi+3_4+2",
        "theta+3+4_vs_2",
        "theta+3_4+2",
    }
