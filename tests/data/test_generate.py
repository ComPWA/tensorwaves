import expertsystem.amplitude.model as es
import numpy as np
import pytest

from tensorwaves.data.generate import generate_data, generate_phsp
from tensorwaves.physics.helicity_formalism.amplitude import IntensityBuilder
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
def test_generate_phsp(
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
    data_sq = data_sample ** 2
    e_sq = data_sq[:, :, 3]
    p3_sq = data_sq[:, :, :3]
    m_sq = np.abs(e_sq - p3_sq.sum(axis=2))
    assert pytest.approx(list(np.sqrt(m_sq.mean(axis=1))), abs=1e-4) == [
        0,
        0.135,
        0.135,
    ]
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
    assert pytest.approx(data_set["mSq_2"].mean()) == 0
    assert pytest.approx(data_set["mSq_3"].mean()) == data_set["mSq_4"].mean()
    assert pytest.approx(data_set["mSq_3_4"].mean(), abs=1e-1) == 1
