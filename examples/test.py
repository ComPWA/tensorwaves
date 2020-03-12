"""How to great an intensity tree from a YAML file using TensorWaves"""

import yaml
from tensorwaves.physics.helicityformalism.amplitude import create_intensity


def main():
    """Main function"""
    with open("examples/intensity-recipe.yaml") as yaml_file:
        decay_info = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
        intensity = create_intensity(decay_info)
        print(intensity)
        print(intensity({'test': [0.1, 0.2, 1.0, 2.0, 1.5]}))
        intensity.summary()


if __name__ == "__main__":
    main()
