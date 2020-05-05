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

    # now verify particle list
    if isinstance(recipe, list):
        for particle in recipe:
            if particle["Name"] in particles:
                logging.warning(
                    "Multiple definitions of particle %s.", particle["Name"],
                )
            else:
                particles[particle["Name"]] = particle
    else:
        raise LookupError("Could not find ParticleList in file.")

    return particles
