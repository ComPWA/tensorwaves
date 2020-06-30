"""Helper functions for the benchmark."""


__all__ = [
    "create_intensity",
    "create_kinematics",
    "generate_phsp_sample",
    "open_recipe",
]

import logging
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

from expertsystem.amplitude.canonicaldecay import CanonicalAmplitudeGenerator
from expertsystem.amplitude.helicitydecay import HelicityAmplitudeGenerator
from expertsystem.ui.system_control import StateTransitionManager

import numpy as np

import yaml

from tensorwaves.data.generate import (
    generate_data,
    generate_phsp,
)
from tensorwaves.physics.helicityformalism.amplitude import (
    IntensityBuilder,
    IntensityTF,
)
from tensorwaves.physics.helicityformalism.kinematics import HelicityKinematics
from tensorwaves.physics.particle import load_particle_list

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.ERROR)


def create_recipe(
    initial_state: List[Union[str, Tuple]],
    final_state: List[Union[str, Tuple]],
    recipe_file_name: str,
    allowed_intermediate_particles: Optional[List[str]] = None,
    formalism_type: str = "helicity",
) -> None:
    if allowed_intermediate_particles is None:
        allowed_intermediate_particles = []
    stm = StateTransitionManager(
        initial_state=initial_state,
        final_state=final_state,
        allowed_intermediate_particles=allowed_intermediate_particles,
        formalism_type=formalism_type,
        topology_building="isobar",
    )
    graph_interaction_settings_groups = stm.prepare_graphs()
    solutions, _ = stm.find_solutions(graph_interaction_settings_groups)
    if formalism_type == "helicity":
        amplitude_generator = HelicityAmplitudeGenerator()
    elif formalism_type == "canonical-helicity":
        amplitude_generator = CanonicalAmplitudeGenerator()
    amplitude_generator.generate(solutions)
    amplitude_generator.write_to_file(recipe_file_name)


def open_recipe(recipe_file_name: str) -> dict:
    with open(recipe_file_name) as input_file:
        recipe = yaml.load(input_file.read(), Loader=yaml.SafeLoader)
    return recipe


def create_kinematics(recipe_file_name: str) -> HelicityKinematics:
    recipe = open_recipe(recipe_file_name)
    kinematics = HelicityKinematics.from_recipe(recipe)
    return kinematics


def generate_phsp_sample(
    recipe_file_name: str, number_of_events: Union[float, int]
) -> np.ndarray:
    kinematics = create_kinematics(recipe_file_name)
    phsp_sample = generate_phsp(int(number_of_events), kinematics)
    return phsp_sample


def create_intensity(
    recipe_file_name: str, phsp_sample: np.ndarray
) -> IntensityTF:
    recipe = open_recipe(recipe_file_name)
    kinematics = create_kinematics(recipe_file_name)
    particles = load_particle_list(recipe_file_name)
    builder = IntensityBuilder(particles, kinematics, phsp_sample)
    intensity = builder.create_intensity(recipe)
    return intensity


def generate_data_sample(
    recipe_file_name: str,
    number_of_events: Union[float, int],
    kinematics: HelicityKinematics,
    intensity: IntensityTF,
) -> np.ndarray:
    kinematics = create_kinematics(recipe_file_name)
    data_sample = generate_data(int(number_of_events), kinematics, intensity)
    return data_sample
