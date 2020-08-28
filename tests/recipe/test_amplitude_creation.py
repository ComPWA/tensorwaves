import os

from tensorwaves.data.generate import generate_phsp
from tensorwaves.physics.helicity_formalism.amplitude import IntensityBuilder
from tensorwaves.physics.helicity_formalism.kinematics import (
    HelicityKinematics,
)
from tensorwaves.physics.particle import extract_particles

from . import (
    create_recipe,
    open_recipe,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


NUMBER_OF_PHSP_EVENTS = 10


def _create_helicity_recipe():
    recipe_filename = "tests/amplitude_model_heli.yml"
    create_recipe(recipe_filename, formalism="helicity")
    recipe = open_recipe(recipe_filename)
    return recipe


def _get_particle_list(recipe):
    particle_list = extract_particles(recipe)
    return particle_list


def _get_kinematics(recipe):
    kinematics = HelicityKinematics.from_recipe(recipe)
    return kinematics


def _generate_phsp(recipe, number_of_events):
    kinematics = _get_kinematics(recipe)
    phsp_sample = generate_phsp(number_of_events, kinematics)
    return phsp_sample


class TestRecipeHelicity:
    recipe = _create_helicity_recipe()
    particle_list = _get_particle_list(recipe)
    kinematics = _get_kinematics(recipe)
    phsp_sample = _generate_phsp(recipe, NUMBER_OF_PHSP_EVENTS)

    def test_particle_list(self):
        particle_list = extract_particles(self.recipe)
        assert set(particle_list) == {"J/psi(1S)", "f(0)(980)", "gamma", "pi0"}

    def test_kinematics(self):
        kinematics = self.kinematics
        masses_is = kinematics.reaction_kinematics_info.initial_state_masses
        masses_fs = kinematics.reaction_kinematics_info.final_state_masses
        assert masses_is == [3.0969]
        assert masses_fs == [0.0, 0.1349768, 0.1349768]

    def test_phsp(self):
        assert self.phsp_sample.shape == (3, NUMBER_OF_PHSP_EVENTS, 4)

    def test_intensity(self):
        builder = IntensityBuilder(
            self.particle_list, self.kinematics, self.phsp_sample
        )
        intensity = builder.create_intensity(self.recipe)
        assert len(intensity.parameters) == 8


def test_canonical():
    recipe_filename = "tests/amplitude_model_cano.yml"
    create_recipe(recipe_filename, formalism="canonical-helicity")
    recipe = open_recipe(recipe_filename)
    particle_list = _get_particle_list(recipe)
    kinematics = _get_kinematics(recipe)
    phsp_sample = _generate_phsp(recipe, NUMBER_OF_PHSP_EVENTS)
    recipe_filename = "tests/amplitude_model_cano.yml"
    builder = IntensityBuilder(particle_list, kinematics, phsp_sample)
    intensity = builder.create_intensity(recipe)
    assert len(intensity.parameters) == 12
