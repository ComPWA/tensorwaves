"""Simple example that shows the workflow of `tensorwaves`."""

import logging
from typing import Tuple

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import yaml

from tensorwaves.data.generate import generate_data, generate_phsp
from tensorwaves.estimator import UnbinnedNLL
from tensorwaves.optimizer.minuit import Minuit2
from tensorwaves.physics.helicityformalism.amplitude import (
    IntensityBuilder,
    IntensityTF,
)
from tensorwaves.physics.helicityformalism.kinematics import HelicityKinematics
from tensorwaves.physics.particle import load_particle_list

logging.getLogger().setLevel(logging.INFO)


def create_kinematics_and_intensity(
    recipe_file_name: str,
) -> Tuple[HelicityKinematics, IntensityTF]:
    with open(recipe_file_name) as input_file:
        recipe = yaml.load(input_file.read(), Loader=yaml.SafeLoader)

    kinematics = HelicityKinematics.from_recipe(recipe)
    part_list = load_particle_list("benchmarks/intensity-recipe.yaml")

    phsp_sample = generate_phsp(300000, kinematics)

    builder = IntensityBuilder(part_list, kinematics, phsp_sample)
    intensity = builder.create_intensity(recipe)
    return kinematics, intensity


def create_estimator(
    kinematics: HelicityKinematics,
    intensity: IntensityTF,
    data_sample: np.ndarray,
) -> None:
    dataset = kinematics.convert(data_sample)

    data_frame = pd.DataFrame(dataset)
    plt.hist(data_frame["mSq_3_4"], bins=100)
    plt.show()

    estimator = UnbinnedNLL(intensity, dataset)

    free_params = {
        # 'Mass_f2(1270):0': 1.3,
        "Width_f2(1270)": 0.3,
        "Mass_f2(1950)": 1.9,
        "Width_f2(1950)": 0.1,
        # 'Mass_f0(980)': 0.8,
        "Width_f0(980)": 0.2,
        "Mass_f0(1500)": 1.6,
        "Width_f0(1500)": 0.01,
        # 'Magnitude_J/psi_to_f2(1270)_0+gamma_-1;f2(1270)_to_pi0_0+pi0_0;': ,
        # 'Phase_J/psi_to_f2(1270)_0+gamma_-1;f2(1270)_to_pi0_0+pi0_0;': ,
        "Magnitude_J/psi_to_f2(1950)_0+gamma_-1;f2(1950)_to_pi0_0+pi0_0;": 3.0,
        "Phase_J/psi_to_f2(1950)_0+gamma_-1;f2(1950)_to_pi0_0+pi0_0;": 1.3,
        "Magnitude_J/psi_to_f2(1270)_-1+gamma_-1;f2(1270)_to_pi0_0+pi0_0;": 6.2,
        "Phase_J/psi_to_f2(1270)_-1+gamma_-1;f2(1270)_to_pi0_0+pi0_0;": 0.71,
        "Magnitude_J/psi_to_f2(1950)_-1+gamma_-1;f2(1950)_to_pi0_0+pi0_0;": 0.53,
        "Phase_J/psi_to_f2(1950)_-1+gamma_-1;f2(1950)_to_pi0_0+pi0_0;": -0.36,
        "Magnitude_J/psi_to_f2(1270)_-2+gamma_-1;f2(1270)_to_pi0_0+pi0_0;": 5.1,
        "Phase_J/psi_to_f2(1270)_-2+gamma_-1;f2(1270)_to_pi0_0+pi0_0;": 2.3,
        "Magnitude_J/psi_to_f2(1950)_-2+gamma_-1;f2(1950)_to_pi0_0+pi0_0;": 0.78,
        "Phase_J/psi_to_f2(1950)_-2+gamma_-1;f2(1950)_to_pi0_0+pi0_0;": -0.52,
        "Magnitude_J/psi_to_f0(980)_0+gamma_-1;f0(980)_to_pi0_0+pi0_0;": 3.4,
        "Phase_J/psi_to_f0(980)_0+gamma_-1;f0(980)_to_pi0_0+pi0_0;": -0.95,
        "Magnitude_J/psi_to_f0(1500)_0+gamma_-1;f0(1500)_to_pi0_0+pi0_0;": 3.2,
        "Phase_J/psi_to_f0(1500)_0+gamma_-1;f0(1500)_to_pi0_0+pi0_0;": 0.59,
    }

    params = {}
    for name in estimator.parameters:
        if name in free_params:
            params[name] = free_params[name]
    logging.info(params)

    logging.info("starting fit")
    minuit2 = Minuit2()
    result = minuit2.optimize(estimator, params)
    logging.info(result)


def main() -> None:
    kinematics, intensity = create_kinematics_and_intensity(
        "benchmarks/intensity-recipe.yaml"
    )
    data_sample = generate_data(30000, kinematics, intensity)
    print(data_sample)


if __name__ == "__main__":
    main()
