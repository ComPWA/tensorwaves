"""Phase space generation using tensorflow."""

from typing import Optional, Union

import numpy as np
import phasespace
import tensorflow as tf
from phasespace.random import SeedLike, get_rng

from tensorwaves.interfaces import (
    PhaseSpaceGenerator,
    UniformRealNumberGenerator,
)
from tensorwaves.physics.helicity_formalism.kinematics import (
    ParticleReactionKinematicsInfo,
)


class TFPhaseSpaceGenerator(PhaseSpaceGenerator):
    """Implements a phase space generator using tensorflow."""

    def __init__(
        self, reaction_kinematics_info: ParticleReactionKinematicsInfo
    ) -> None:
        self.phsp_gen = phasespace.nbody_decay(
            reaction_kinematics_info.total_invariant_mass,
            reaction_kinematics_info.final_state_masses,
        )

    def generate(
        self,
        size: int,
        seed_or_rng: Optional[Union[UniformRealNumberGenerator, int]] = None,
    ) -> np.ndarray:
        seed = None
        if isinstance(seed_or_rng, UniformRealNumberGenerator):
            seed = seed_or_rng.seed
        elif isinstance(seed_or_rng, int):
            seed = seed_or_rng
        else:
            raise NotImplementedError(
                f"Cannot work with a seed or RNG of type {seed_or_rng.__class__.__name__}"
            )
        weights, particles = self.phsp_gen.generate(n_events=size, seed=seed)
        particles = np.array(
            tuple(particles[x].numpy() for x in particles.keys())
        )
        return particles, weights.numpy()


class TFUniformRealNumberGenerator(UniformRealNumberGenerator):
    """Implements a uniform real random number generator using tensorflow."""

    def __init__(self, seed: SeedLike = None):
        self.seed = seed
        self.dtype = tf.float64

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
    def seed(self) -> SeedLike:
        return self.__seed

    @seed.setter
    def seed(self, value: SeedLike) -> None:
        self.__seed = value
        self.generator = get_rng(self.seed)
