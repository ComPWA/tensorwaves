# pylint: disable=import-outside-toplevel
"""Implementations of `.FourMomentumGenerator`."""

from typing import Mapping, Tuple

import numpy as np

from tensorwaves.interface import (
    DataSample,
    FourMomentumGenerator,
    RealNumberGenerator,
)

from .rng import TFUniformRealNumberGenerator


class TFPhaseSpaceGenerator(FourMomentumGenerator):
    """Implements a phase space generator using tensorflow.

    Args:
        initial_state_mass: Mass of the decaying state.
        final_state_masses: A mapping of final state IDs to the corresponding
            masses.
    """

    def __init__(
        self,
        initial_state_mass: float,
        final_state_masses: Mapping[int, float],
    ) -> None:
        import phasespace

        sorted_ids = sorted(final_state_masses)
        self.__phsp_gen = phasespace.nbody_decay(
            mass_top=initial_state_mass,
            masses=[final_state_masses[i] for i in sorted_ids],
            names=list(map(str, sorted_ids)),
        )

    def generate(
        self, size: int, rng: RealNumberGenerator
    ) -> Tuple[DataSample, np.ndarray]:
        if not isinstance(rng, TFUniformRealNumberGenerator):
            raise TypeError(
                f"{TFPhaseSpaceGenerator.__name__} requires a "
                f"{TFUniformRealNumberGenerator.__name__}, but fed a "
                f"{type(rng).__name__}"
            )
        weights, particles = self.__phsp_gen.generate(
            n_events=size, seed=rng.generator
        )
        phsp_momenta = {
            f"p{label}": momenta.numpy()[:, [3, 0, 1, 2]]
            for label, momenta in particles.items()
        }
        return phsp_momenta, weights.numpy()
