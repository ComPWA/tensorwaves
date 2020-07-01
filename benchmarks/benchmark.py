"""Sketch of a general fit procedure, used as quick benchmark check."""

from os.path import dirname, realpath

from tensorwaves.estimator import UnbinnedNLL
from tensorwaves.optimizer.minuit import Minuit2

from . import (
    create_intensity,
    create_kinematics,
    create_recipe,
    generate_data_sample,
    generate_phsp_sample,
)


SCRIPT_DIR = dirname(realpath(__file__))
RECIPE_FILE = f"{SCRIPT_DIR}/intensity-recipe.yaml"

INITIAL_STATE = [("J/psi")]
FINAL_STATE = [("gamma"), ("pi0"), ("pi0")]
INTERMEDIATE_STATES = ["f0(980)"]

FORMALISM_TYPE: str = "helicity"

NUMBER_OF_PHSP_EVENTS = 3e5
NUMBER_OF_DATA_EVENTS = 3e4


def run_benchmark() -> None:
    create_recipe(
        initial_state=INITIAL_STATE,
        final_state=FINAL_STATE,
        allowed_intermediate_particles=INTERMEDIATE_STATES,
        recipe_file_name=RECIPE_FILE,
        formalism_type=FORMALISM_TYPE,
    )
    kinematics = create_kinematics(RECIPE_FILE)
    phsp_sample = generate_phsp_sample(RECIPE_FILE, NUMBER_OF_PHSP_EVENTS)
    intensity = create_intensity(RECIPE_FILE, phsp_sample)
    data_sample = generate_data_sample(
        recipe_file_name=RECIPE_FILE,
        number_of_events=NUMBER_OF_DATA_EVENTS,
        kinematics=kinematics,
        intensity=intensity,
    )
    data_set = kinematics.convert(data_sample)
    estimator = UnbinnedNLL(intensity, data_set)
    initial_parameters = estimator.parameters
    minuit2 = Minuit2()
    minuit2.optimize(estimator, initial_parameters)


if __name__ == "__main__":
    run_benchmark()
