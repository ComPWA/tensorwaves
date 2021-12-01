# pylint: disable=import-outside-toplevel
"""Implementations of `.PhaseSpaceGenerator` and `.UniformRealNumberGenerator`."""

from typing import Mapping, Optional, Tuple

import numpy as np

from tensorwaves.interface import (
    DataSample,
    PhaseSpaceGenerator,
    UniformRealNumberGenerator,
)


class TFPhaseSpaceGenerator(PhaseSpaceGenerator):
    """Implements a phase space generator using tensorflow."""

    def __init__(self) -> None:
        self.__phsp_gen = None

    def setup(
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
        self, size: int, rng: UniformRealNumberGenerator
    ) -> Tuple[DataSample, np.ndarray]:
        if not isinstance(rng, TFUniformRealNumberGenerator):
            raise TypeError(
                f"{TFPhaseSpaceGenerator.__name__} requires a "
                f"{TFUniformRealNumberGenerator.__name__}, but fed a "
                f"{rng.__class__.__name__}"
            )
        if self.__phsp_gen is None:
            raise ValueError("Phase space generator has not been set up")
        weights, particles = self.__phsp_gen.generate(
            n_events=size, seed=rng.generator
        )
        phsp_sample = {
            int(label): momenta.numpy()[:, [3, 0, 1, 2]]
            for label, momenta in particles.items()
        }
        return phsp_sample, weights.numpy()


class TFUniformRealNumberGenerator(UniformRealNumberGenerator):
    """Implements a uniform real random number generator using tensorflow."""

    def __init__(self, seed: Optional[float] = None):
        from tensorflow import float64

        self.seed = seed
        self.dtype = float64

    def __call__(
        self, size: int, min_value: float = 0.0, max_value: float = 1.0
    ) -> np.ndarray:
        return self.generator.uniform(
            shape=[size],
            minval=min_value,
            maxval=max_value,
            dtype=self.dtype,
        ).numpy()

    @property
    def seed(self) -> Optional[float]:
        return self.__seed

    @seed.setter
    def seed(self, value: Optional[float]) -> None:
        from phasespace.random import get_rng

        self.__seed = value
        self.generator = get_rng(self.seed)
