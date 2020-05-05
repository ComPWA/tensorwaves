import yaml

from tensorwaves.data.generate import generate_data, generate_phsp
from tensorwaves.data.tf_phasespace import (
    TFPhaseSpaceGenerator,
    TFUniformRealNumberGenerator,
)
from tensorwaves.physics.particle import load_particle_list
from tensorwaves.physics.helicityformalism.amplitude import IntensityBuilder
from tensorwaves.physics.helicityformalism.kinematics import HelicityKinematics


JPSI_MASS = 3.096900
PI0_MASS = 0.1349766


with open("examples/intensity-recipe.yaml") as fc:
    recipe = yaml.load(fc.read(), Loader=yaml.SafeLoader)
    kin = HelicityKinematics.from_recipe(recipe)
    part_list = load_particle_list("examples/intensity-recipe.yaml")

    phsp_sample = generate_phsp(300000, kin)

    builder = IntensityBuilder(part_list, kin, phsp_sample)
    intensity = builder.create_intensity(recipe)

    data_sample = generate_data(30000, intensity, kin)

    dataset = kin.convert(data_sample)

    # ploting
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame(dataset)
    plt.hist(df["mSq_3_4"], bins=100)
    plt.show()
