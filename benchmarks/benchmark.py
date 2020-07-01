"""Sketch of a general fit procedure, used as quick benchmark check."""

import logging
import timeit
from os.path import dirname, realpath
from typing import (
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from typing import NamedTuple

from expertsystem.amplitude.canonicaldecay import CanonicalAmplitudeGenerator
from expertsystem.amplitude.helicitydecay import HelicityAmplitudeGenerator
from expertsystem.ui.system_control import StateTransitionManager

import yaml

from tensorwaves.data.generate import (
    generate_data,
    generate_phsp,
)
from tensorwaves.estimator import UnbinnedNLL
from tensorwaves.optimizer.minuit import Minuit2
from tensorwaves.physics.helicityformalism.amplitude import IntensityBuilder
from tensorwaves.physics.helicityformalism.kinematics import HelicityKinematics
from tensorwaves.physics.particle import load_particle_list

logging.disable(logging.WARNING)
logging.getLogger().setLevel(logging.ERROR)


SCRIPT_DIR = dirname(realpath(__file__))
RECIPE_FILE = f"{SCRIPT_DIR}/intensity-recipe.yml"

INITIAL_STATE = [("J/psi")]
FINAL_STATE = [("gamma"), ("pi0"), ("pi0")]
INTERMEDIATE_STATES = ["f0(980)"]

FORMALISM_TYPE: str = "helicity"

NUMBER_OF_PHSP_EVENTS = 3e5
NUMBER_OF_DATA_EVENTS = 3e4


def run_benchmark() -> None:  # pylint: disable=too-many-locals
    create_recipe(
        initial_state=INITIAL_STATE,
        final_state=FINAL_STATE,
        allowed_intermediate_particles=INTERMEDIATE_STATES,
        recipe_file_name=RECIPE_FILE,
        formalism_type=FORMALISM_TYPE,
    )
    with open(RECIPE_FILE) as input_file:
        recipe = yaml.load(input_file.read(), Loader=yaml.SafeLoader)

    # Create phase space sample
    kinematics = HelicityKinematics.from_recipe(recipe)
    phsp_timer = timeit.default_timer()
    phsp_sample = generate_phsp(int(NUMBER_OF_PHSP_EVENTS), kinematics)
    phsp_timer = timeit.default_timer() - phsp_timer

    # Create intensity-based sample
    particles = load_particle_list(RECIPE_FILE)
    builder = IntensityBuilder(particles, kinematics, phsp_sample)
    intensity = builder.create_intensity(recipe)
    data_timer = timeit.default_timer()
    data_sample = generate_data(
        int(NUMBER_OF_DATA_EVENTS), kinematics, intensity
    )
    data_timer = timeit.default_timer() - data_timer

    # Optimize intensity
    data_set = kinematics.convert(data_sample)
    estimator = UnbinnedNLL(intensity, data_set)
    initial_parameters = estimator.parameters
    minuit2 = Minuit2()
    fit_timer = timeit.default_timer()
    minuit2.optimize(estimator, initial_parameters)
    fit_timer = timeit.default_timer()

    # Print output
    color = Color()
    print(color.bold)
    print("Particle decay:")
    print("  Initial state:", INITIAL_STATE)
    print("  Final state:", FINAL_STATE)
    print("  Intermediate states:", INTERMEDIATE_STATES)
    print("Durations:")
    print(f"  1. phsp generation: {phsp_timer:.2f}s")
    print(f"  2. data generation: {data_timer:.2f}s")
    print(f"  3. fit generation:  {fit_timer:.2f}s")
    print(color.end, end="")


def create_recipe(
    initial_state: Sequence[Union[str, Tuple]],
    final_state: Sequence[Union[str, Tuple]],
    recipe_file_name: str,
    allowed_intermediate_particles: Optional[List[str]] = None,
    formalism_type: str = "helicity",
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


class Color(NamedTuple):
    """Terminal colors."""

    ok: str = "\033[92m"
    warning: str = "\033[93m"
    fail: str = "\033[91m"
    bold: str = "\033[1m"
    underline: str = "\033[4m"
    end: str = "\033[0m"


if __name__ == "__main__":
    run_benchmark()
