import logging

import yaml

from tensorwaves.data.generate import generate_data, generate_phsp
from tensorwaves.estimator import UnbinnedNLL
from tensorwaves.physics.helicityformalism.amplitude import IntensityBuilder
from tensorwaves.physics.helicityformalism.kinematics import HelicityKinematics
from tensorwaves.physics.particle import load_particle_list
from tensorwaves.optimizer.minuit import Minuit2

logging.getLogger().setLevel(logging.INFO)

with open("examples/intensity-recipe.yaml") as fc:
    recipe = yaml.load(fc.read(), Loader=yaml.SafeLoader)
    kin = HelicityKinematics.from_recipe(recipe)
    part_list = load_particle_list("examples/intensity-recipe.yaml")

    phsp_sample = generate_phsp(300000, kin)

    builder = IntensityBuilder(part_list, kin, phsp_sample)
    intensity = builder.create_intensity(recipe)

    data_sample = generate_data(30000, kin, intensity)

    dataset = kin.convert(data_sample)

    # plotting
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame(dataset)
    plt.hist(df["mSq_3_4"], bins=100)
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
    for name, value in estimator.parameters.items():
        if name in free_params:
            params[name] = free_params[name]
    logging.info(params)

    logging.info("starting fit")
    minu2 = Minuit2()
    result = minu2.optimize(estimator, params)
    logging.info(result)
