"""Required to set mypy options for the tests folder."""

import logging

import yaml
from expertsystem.amplitude.canonical_decay import CanonicalAmplitudeGenerator
from expertsystem.amplitude.helicity_decay import HelicityAmplitudeGenerator
from expertsystem.ui import InteractionTypes, StateTransitionManager

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.ERROR)


def create_recipe(filename: str, formalism: str = "helicity") -> None:
    stm = StateTransitionManager(
        initial_state=[("J/psi(1S)")],
        final_state=[("gamma"), ("pi0"), ("pi0")],
        allowed_intermediate_particles=["f(0)(980)"],
        formalism_type=formalism,
        topology_building="isobar",
    )

    stm.set_allowed_interaction_types([InteractionTypes.Strong])
    graph_interaction_settings_groups = stm.prepare_graphs()
    solutions, _ = stm.find_solutions(graph_interaction_settings_groups)

    if formalism == "helicity":
        amplitude_generator = HelicityAmplitudeGenerator()
    elif formalism == "canonical-helicity":
        amplitude_generator = CanonicalAmplitudeGenerator()
    else:
        raise NotImplementedError(f"No amplitude generator for {formalism}")

    amplitude_generator.generate(solutions)
    amplitude_generator.write_to_file(filename)


def open_recipe(filename: str) -> dict:
    with open(filename) as input_file:
        recipe = yaml.load(input_file.read(), Loader=yaml.SafeLoader)
    return recipe
