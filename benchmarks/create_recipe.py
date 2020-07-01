"""Generate amplitude model recipe files that can be used by the benchmark.

This script was used to generate the YAML recipe files that came with the
benchmark tests. For more info, see the `expertsystem usage page
<https://pwa.readthedocs.io/projects/expertsystem/en/latest/usage.html>`.
"""

import logging
from os.path import dirname, realpath
from typing import (
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from expertsystem.amplitude.canonicaldecay import CanonicalAmplitudeGenerator
from expertsystem.amplitude.helicitydecay import HelicityAmplitudeGenerator
from expertsystem.ui.system_control import (
    InteractionTypes,
    StateTransitionManager,
)

logging.disable(logging.WARNING)
logging.getLogger().setLevel(logging.ERROR)


SCRIPT_DIR = dirname(realpath(__file__))

CASES = [
    {
        "output_file": "Jpsi_f0_gamma.pi0.pi0_heli.yml",
        "initial_state": [("J/psi")],
        "final_state": [("gamma"), ("pi0"), ("pi0")],
        "intermediate_states": ["f0"],
        "interaction_types": [InteractionTypes.Strong],
        "formalism_type": "helicity",
    },
    {
        "output_file": "Jpsi_f0_gamma.pi0.pi0_cano.yml",
        "initial_state": [("J/psi")],
        "final_state": [("gamma"), ("pi0"), ("pi0")],
        "intermediate_states": ["f0"],
        "interaction_types": [InteractionTypes.Strong],
        "formalism_type": "canonical-helicity",
    },
    {
        "output_file": "Jpsi_f0.omega_gamma.pi0.pi0_heli.yml",
        "initial_state": [("J/psi")],
        "final_state": [("gamma"), ("pi0"), ("pi0")],
        "intermediate_states": ["f0", "omega"],
        "formalism_type": "helicity",
    },
]


def create_recipe(
    initial_state: Sequence[Union[str, Tuple]],
    final_state: Sequence[Union[str, Tuple]],
    recipe_file_name: str,
    formalism_type: str,
    allowed_intermediate_particles: Optional[List[str]] = None,
    interaction_types: Optional[InteractionTypes] = None,
) -> None:
    if allowed_intermediate_particles is None:
        allowed_intermediate_particles = []
    stm = StateTransitionManager(
        initial_state=list(initial_state),
        final_state=list(final_state),
        allowed_intermediate_particles=allowed_intermediate_particles,
        formalism_type=formalism_type,
        topology_building="isobar",
    )
    if interaction_types:
        stm.set_allowed_interaction_types(interaction_types)
    graph_interaction_settings_groups = stm.prepare_graphs()
    solutions, _ = stm.find_solutions(graph_interaction_settings_groups)
    if formalism_type == "helicity":
        amplitude_generator = HelicityAmplitudeGenerator()
    elif formalism_type == "canonical-helicity":
        amplitude_generator = CanonicalAmplitudeGenerator()
    else:
        raise Exception(f"Formalism '{formalism_type}' cannot be handled")
    amplitude_generator.generate(solutions)
    amplitude_generator.write_to_file(recipe_file_name)


def create_recipes() -> None:
    for i, case in enumerate(CASES, 1):
        output_file = str(case["output_file"])
        print()
        print(f"\033[1m--== Case {i}/{len(CASES)}: {output_file} ==--\033[0m")
        interaction_types = case.get("interaction_types", None)
        if interaction_types is not None:
            interaction_types = list(interaction_types)
        create_recipe(
            initial_state=list(case["initial_state"]),
            final_state=list(case["final_state"]),
            allowed_intermediate_particles=list(case["intermediate_states"]),
            recipe_file_name=f"{SCRIPT_DIR}/{output_file}",
            formalism_type=str(case["formalism_type"]),
            interaction_types=interaction_types,
        )


if __name__ == "__main__":
    create_recipes()
