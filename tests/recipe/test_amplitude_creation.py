from tensorwaves.data.generate import generate_phsp
from tensorwaves.physics.helicityformalism.kinematics import HelicityKinematics
from tensorwaves.physics.particle import extract_particles

from . import (
    create_recipe,
    open_recipe,
)


def _create_helicity_recipe():
    recipe_filename = "tests/amplitude_model_heli.yml"
    create_recipe(recipe_filename, formalism="helicity")
    recipe = open_recipe(recipe_filename)
    return recipe


class TestRecipeHelicity:
    recipe = _create_helicity_recipe()
    number_of_phsp_events = 10

    def test_particle_list(self):
        particle_list = extract_particles(self.recipe)
        assert set(particle_list) == {"J/psi", "f0(980)", "gamma", "pi0"}

    def test_kinematics(self):
        kinematics = HelicityKinematics.from_recipe(self.recipe)
        masses_is = kinematics.reaction_kinematics_info.initial_state_masses
        masses_fs = kinematics.reaction_kinematics_info.final_state_masses
        assert masses_is == [3.096900]
        assert masses_fs == [0.0, 0.1349766, 0.1349766]

    def generate_phsp(self):
        kinematics = HelicityKinematics.from_recipe(self.recipe)
        phsp_sample = generate_phsp(self.number_of_phsp_events, kinematics)
        return phsp_sample

    def test_phsp(self):
        phsp_sample = self.generate_phsp()
        assert phsp_sample.shape == (3, self.number_of_phsp_events, 4)

    def test_helicity(self):
        assert len(self.recipe) == 5


def test_canonical():
    recipe_filename = "tests/amplitude_model_cano.yml"
    create_recipe(recipe_filename, formalism="canonical-helicity")
    recipe = open_recipe(recipe_filename)
    assert len(recipe) == 5
