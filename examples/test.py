import xmltodict
from yaml import load
from tensorwaves.physics.helicityformalism.amplitude import create_intensity

with open("examples/intensity-recipe.yaml") as fc:
    decay_info = load(fc.read())
    intensity = create_intensity(decay_info)
    print(intensity)
    print(intensity({"test": [0.1, 0.2, 1.0, 2.0, 1.5]}))
    intensity.summary()
