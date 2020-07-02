"""Sketch of a general fit procedure, used as quick benchmark check."""

import argparse
import logging
import os.path
import timeit
from typing import (
    Any,
    Dict,
    List,
)
from typing import NamedTuple

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


def run_benchmark() -> None:  # pylint: disable=too-many-locals

    # Get terminal arguments
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "recipe_file",
        help="YAML amplitude model recipe file, produced with the expertsystem",
    )
    parser.add_argument("--phsp", help="Number of phase space events")
    parser.add_argument("--data", help="Number of intensity-based events")
    args = parser.parse_args()
    number_of_data_events = int(3e3)
    if args.data:
        number_of_data_events = int(float(args.data))
    number_of_phsp_events = int(3e4)
    if args.phsp:
        number_of_phsp_events = int(float(args.phsp))

    # Open recipe file
    input_recipe_file = args.recipe_file
    with open(input_recipe_file) as input_file:
        recipe = yaml.load(input_file.read(), Loader=yaml.SafeLoader)
    free_parameters = get_free_parameters(recipe)

    # Create phase space sample
    kinematics = HelicityKinematics.from_recipe(recipe)
    phsp_timer = timeit.default_timer()
    phsp_sample = generate_phsp(int(number_of_phsp_events), kinematics)
    phsp_timer = timeit.default_timer() - phsp_timer

    # Create intensity-based sample
    particles = load_particle_list(input_recipe_file)
    builder = IntensityBuilder(particles, kinematics, phsp_sample)
    intensity = builder.create_intensity(recipe)
    data_timer = timeit.default_timer()
    data_sample = generate_data(
        int(number_of_data_events), kinematics, intensity
    )
    data_timer = timeit.default_timer() - data_timer

    # Define estimator and parameters
    data_set = kinematics.convert(data_sample)
    estimator = UnbinnedNLL(intensity, data_set)

    # Optimize model
    minuit2 = Minuit2()
    fit_timer = timeit.default_timer()
    minuit2.optimize(estimator, free_parameters)
    fit_timer = timeit.default_timer()

    # Print output
    color = Color()
    print(color.bold)
    print("Recipe file:", os.path.realpath(args.recipe_file))
    print("Number of free parameters:", len(free_parameters))
    print("Number of events:")
    print("  - Phase space:    ", number_of_phsp_events)
    print("  - Intensity-based:", number_of_data_events)
    print("Durations:")
    print(f"  1. phsp generation: {phsp_timer:.2f}s")
    print(f"  2. data generation: {data_timer:.2f}s")
    print(f"  3. fit generation:  {fit_timer:.2f}s")
    print(color.end, end="")


def get_free_parameters(
    recipe: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, float]:
    parameter_list = recipe.get("Parameters", None)
    if parameter_list is None:
        raise Exception('Recipe file does not contain "Parameters" section!')
    free_parameters = {
        str(item["Name"]): item["Value"]
        for item in parameter_list
        if not item.get("Fix", False)
    }
    return free_parameters


class Color(NamedTuple):
    """Terminal print colors."""

    ok: str = "\033[92m"
    warning: str = "\033[93m"
    fail: str = "\033[91m"
    bold: str = "\033[1m"
    underline: str = "\033[4m"
    end: str = "\033[0m"


if __name__ == "__main__":
    run_benchmark()
