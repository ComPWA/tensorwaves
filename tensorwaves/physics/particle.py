"""Collection of helper functions for particles."""

import logging

import yaml


def load_particle_list(particle_list_file: str) -> dict:
    """Load particle list from a yaml file."""
    with open(particle_list_file) as configfile:
        fullconfig = yaml.load(configfile.read(), Loader=yaml.SafeLoader)
        return extract_particles(fullconfig)


def extract_particles(recipe: dict) -> dict:
    if "ParticleList" in recipe:
        recipe = recipe["ParticleList"]

    particles = dict()

    # now verify particle dict
    if isinstance(recipe, dict):
        for name, particle in recipe.items():
            if NameError in particles:
                logging.warning(
                    "Multiple definitions of particle %s.", name,
                )
            else:
                particles[name] = particle
    else:
        raise LookupError("Could not find ParticleList in file.")

    return particles
