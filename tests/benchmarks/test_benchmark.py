r"""Benchmark for :math:`J/\psi \to f_0(*)\gamma \to \pi^0\pi^0\gamma`."""

# pylint: disable=redefined-outer-name

from os.path import dirname, realpath
from typing import Union

import numpy as np

import pytest

from tensorwaves.physics.helicityformalism.amplitude import IntensityTF
from tensorwaves.physics.helicityformalism.kinematics import HelicityKinematics

from . import (
    create_intensity,
    create_kinematics,
    create_recipe,
    generate_data_sample,
    generate_phsp_sample,
)


SCRIPT_DIR = dirname(realpath(__file__))

RECIPE_FILE = f"{SCRIPT_DIR}/intensity-recipe.yaml"
NUMBER_OF_PHSP_EVENTS = 3e5
NUMBER_OF_DATA_EVENTS = 3e4


@pytest.mark.slow
@pytest.mark.parametrize(
    "initial_state, final_state, intermediate_particles",
    [([("J/psi")], [("gamma"), ("pi0"), ("pi0")], ["f0(980)"])],
)
def test_benchmark(
    benchmark, initial_state, final_state, intermediate_particles
):  # type: ignore
    @benchmark
    def generate_phsp(
        recipe_file: str, number_of_events: Union[float, int]
    ) -> np.ndarray:
        return generate_phsp_sample(recipe_file, number_of_events)

    @benchmark
    def benchmark_generate_data(
        recipe_file_name: str,
        number_of_events: Union[float, int],
        kinematics: HelicityKinematics,
        intensity: IntensityTF,
    ) -> np.ndarray:
        return generate_data_sample(
            recipe_file_name=recipe_file_name,
            number_of_events=number_of_events,
            kinematics=kinematics,
            intensity=intensity,
        )

    create_recipe(
        initial_state=initial_state,
        final_state=final_state,
        allowed_intermediate_particles=intermediate_particles,
        recipe_file_name=RECIPE_FILE,
        formalism_type="helicity",
    )
    kinematics = create_kinematics(RECIPE_FILE)
    phsp_sample = generate_phsp(RECIPE_FILE, NUMBER_OF_PHSP_EVENTS)
    intensity = create_intensity(RECIPE_FILE, phsp_sample)
    benchmark_generate_data(
        recipe_file_name=RECIPE_FILE,
        number_of_events=NUMBER_OF_DATA_EVENTS,
        kinematics=kinematics,
        intensity=intensity,
    )

    # def create_estimator(
    #     kinematics: HelicityKinematics,
    #     intensity: IntensityTF,
    #     data_sample: np.ndarray,
    # ) -> None:
    #     dataset = kinematics.convert(data_sample)

    #     data_frame = pd.DataFrame(dataset)
    #     plt.hist(data_frame["mSq_3_4"], bins=100)
    #     plt.show()

    #     estimator = UnbinnedNLL(intensity, dataset)

    #     free_params = {
    #         # 'Mass_f2(1270):0': 1.3,
    #         "Width_f2(1270)": 0.3,
    #         "Mass_f2(1950)": 1.9,
    #         "Width_f2(1950)": 0.1,
    #         # 'Mass_f0(980)': 0.8,
    #         "Width_f0(980)": 0.2,
    #         "Mass_f0(1500)": 1.6,
    #         "Width_f0(1500)": 0.01,
    #         # 'Magnitude_J/psi_to_f2(1270)_0+gamma_-1;f2(1270)_to_pi0_0+pi0_0;': ,
    #         # 'Phase_J/psi_to_f2(1270)_0+gamma_-1;f2(1270)_to_pi0_0+pi0_0;': ,
    #         "Magnitude_J/psi_to_f2(1950)_0+gamma_-1;f2(1950)_to_pi0_0+pi0_0;": 3.0,
    #         "Phase_J/psi_to_f2(1950)_0+gamma_-1;f2(1950)_to_pi0_0+pi0_0;": 1.3,
    #         "Magnitude_J/psi_to_f2(1270)_-1+gamma_-1;f2(1270)_to_pi0_0+pi0_0;": 6.2,
    #         "Phase_J/psi_to_f2(1270)_-1+gamma_-1;f2(1270)_to_pi0_0+pi0_0;": 0.71,
    #         "Magnitude_J/psi_to_f2(1950)_-1+gamma_-1;f2(1950)_to_pi0_0+pi0_0;": 0.53,
    #         "Phase_J/psi_to_f2(1950)_-1+gamma_-1;f2(1950)_to_pi0_0+pi0_0;": -0.36,
    #         "Magnitude_J/psi_to_f2(1270)_-2+gamma_-1;f2(1270)_to_pi0_0+pi0_0;": 5.1,
    #         "Phase_J/psi_to_f2(1270)_-2+gamma_-1;f2(1270)_to_pi0_0+pi0_0;": 2.3,
    #         "Magnitude_J/psi_to_f2(1950)_-2+gamma_-1;f2(1950)_to_pi0_0+pi0_0;": 0.78,
    #         "Phase_J/psi_to_f2(1950)_-2+gamma_-1;f2(1950)_to_pi0_0+pi0_0;": -0.52,
    #         "Magnitude_J/psi_to_f0(980)_0+gamma_-1;f0(980)_to_pi0_0+pi0_0;": 3.4,
    #         "Phase_J/psi_to_f0(980)_0+gamma_-1;f0(980)_to_pi0_0+pi0_0;": -0.95,
    #         "Magnitude_J/psi_to_f0(1500)_0+gamma_-1;f0(1500)_to_pi0_0+pi0_0;": 3.2,
    #         "Phase_J/psi_to_f0(1500)_0+gamma_-1;f0(1500)_to_pi0_0+pi0_0;": 0.59,
    #     }

    #     params = {}
    #     for name in estimator.parameters:
    #         if name in free_params:
    #             params[name] = free_params[name]
    #     logging.info(params)

    #     logging.info("starting fit")
    #     minuit2 = Minuit2()
    #     result = minuit2.optimize(estimator, params)
    #     logging.info(result)
