"""Evaluateable physics models for amplitude analysis."""

__all__ = ["helicityformalism"]

import logging

import yaml

from . import helicityformalism


def load_particle_list(particle_list_file: str) -> dict:
    """Load particle list from a yaml file."""
    with open(particle_list_file) as configfile:
        particles = dict()
        fullconfig = yaml.load(configfile.read(), Loader=yaml.SafeLoader)

        if "ParticleList" in fullconfig:
            fullconfig = fullconfig["ParticleList"]

        # now verify particle list
        if isinstance(fullconfig, list):
            for particle in fullconfig:
                if particle["Name"] in particles:
                    logging.warning(
                        "Multiple definitions of particle %s.",
                        particle["Name"],
                    )
                else:
                    particles[particle["Name"]] = particle
        else:
            raise LookupError("Could not find ParticleList in file.")

        return particles
