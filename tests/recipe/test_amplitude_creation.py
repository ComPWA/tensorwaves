import os

import expertsystem.amplitude.model as es

from tensorwaves.data.generate import generate_phsp
from tensorwaves.physics.helicity_formalism.amplitude import IntensityBuilder
from tensorwaves.physics.helicity_formalism.kinematics import (
    HelicityKinematics,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


NUMBER_OF_PHSP_EVENTS = 10


def _generate_phsp(recipe: es.AmplitudeModel, number_of_events: int):
    kinematics = HelicityKinematics.from_model(recipe)
    phsp_sample = generate_phsp(number_of_events, kinematics)
    return phsp_sample


def test_helicity(helicity_model: es.AmplitudeModel):
    model = helicity_model
    kinematics = HelicityKinematics.from_model(model)
    masses_is = kinematics.reaction_kinematics_info.initial_state_masses
    masses_fs = kinematics.reaction_kinematics_info.final_state_masses
    assert masses_is == [3.0969]
    assert masses_fs == [0.0, 0.1349768, 0.1349768]

    phsp_sample = _generate_phsp(model, NUMBER_OF_PHSP_EVENTS)
    assert phsp_sample.shape == (3, NUMBER_OF_PHSP_EVENTS, 4)

    builder = IntensityBuilder(model.particles, kinematics, phsp_sample)
    intensity = builder.create_intensity(model)
    assert set(intensity.parameters) == {
        "strength_incoherent",
        "Position_J/psi(1S)",
        "Width_J/psi(1S)",
        "MesonRadius_J/psi(1S)",
        "Magnitude_J/psi(1S)_to_f(0)(980)_0+gamma_1;f(0)(980)_to_pi0_0+pi0_0;",
        "Phase_J/psi(1S)_to_f(0)(980)_0+gamma_1;f(0)(980)_to_pi0_0+pi0_0;",
        "Position_f(0)(980)",
        "Width_f(0)(980)",
        "MesonRadius_f(0)(980)",
    }


def test_canonical(canonical_model: es.AmplitudeModel):
    model = canonical_model
    particles = model.particles
    kinematics = HelicityKinematics.from_model(model)
    phsp_sample = _generate_phsp(model, NUMBER_OF_PHSP_EVENTS)
    builder = IntensityBuilder(particles, kinematics, phsp_sample)
    intensity = builder.create_intensity(model)
    assert set(intensity.parameters) == {
        "strength_incoherent",
        "Position_J/psi(1S)",
        "Width_J/psi(1S)",
        "MesonRadius_J/psi(1S)",
        "Magnitude_J/psi(1S)_to_f(0)(980)_0+gamma_1;f(0)(980)_to_pi0_0+pi0_0;",
        "Phase_J/psi(1S)_to_f(0)(980)_0+gamma_1;f(0)(980)_to_pi0_0+pi0_0;",
        "Position_f(0)(980)",
        "Width_f(0)(980)",
        "MesonRadius_f(0)(980)",
    }
