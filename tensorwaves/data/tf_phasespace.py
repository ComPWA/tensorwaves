"""Phase space generation using tensorflow."""

import numpy as np

import phasespace

import tensorflow as tf

from tensorwaves.interfaces import (
    PhaseSpaceGenerator,
    UniformRealNumberGenerator,
)


class TFPhaseSpaceGenerator(PhaseSpaceGenerator):
    """Implements a phase space generator using tensorflow."""

    def __init__(
        self, initial_state_mass: float, final_state_masses: list
    ) -> None:
        print(phasespace.__file__)
        self.phsp_gen = phasespace.nbody_decay(
            initial_state_mass, final_state_masses
        )

    def generate(
        self, size: int, random_generator: UniformRealNumberGenerator
    ) -> np.ndarray:
        weights, particles = self.phsp_gen.generate(n_events=size)
        particles = np.array(
            tuple(particles[x].numpy() for x in particles.keys())
        )
        return particles, weights


class TFUniformRealNumberGenerator(UniformRealNumberGenerator):
    """Implements a uniform real random number generator using tensorflow."""

    def __init__(self, seed: float):
        self.seed = seed
        self.random = tf.random.uniform
        self.dtype = tf.float64

    def __call__(
        self, size: int, min_value: float = 0.0, max_value: float = 1.0
    ) -> np.ndarray:
        return self.random(
            shape=[size,], minval=min_value, maxval=max_value, dtype=self.dtype
        )

    @property
    def seed(self) -> float:
        return self.__seed

    @seed.setter
    def seed(self, value: float) -> None:
        self.__seed = value
        tf.random.set_seed(self.__seed)
