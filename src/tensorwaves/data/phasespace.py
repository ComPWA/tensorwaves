"""Implementations of `.PhaseSpaceGenerator` and `.UniformRealNumberGenerator`."""

from typing import Optional, Tuple

import ampform as pwa
import numpy as np
import phasespace
import tensorflow as tf
from phasespace.random import get_rng

from tensorwaves.interfaces import (
    MomentumSample,
    PhaseSpaceGenerator,
    UniformRealNumberGenerator,
)


class TFPhaseSpaceGenerator(PhaseSpaceGenerator):
    """Implements a phase space generator using tensorflow."""

    def __init__(self) -> None:
        self.__phsp_gen = None

    def setup(self, reaction_info: pwa.kinematics.ReactionInfo) -> None:
        initial_states = reaction_info.initial_state.values()
        if len(initial_states) != 1:
            raise ValueError("Not a 1-to-n body decay")
        initial_state = next(iter(initial_states))
        self.__phsp_gen = phasespace.nbody_decay(
            mass_top=initial_state.mass,
            masses=[p.mass for p in reaction_info.final_state.values()],
            names=list(map(str, reaction_info.final_state)),
        )

    def generate(
        self, size: int, rng: UniformRealNumberGenerator
    ) -> Tuple[MomentumSample, np.ndarray]:
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
        momentum_pool = {
            int(label): momenta.numpy()[:, [3, 0, 1, 2]]
            for label, momenta in particles.items()
        }
        return momentum_pool, weights.numpy()


class TFUniformRealNumberGenerator(UniformRealNumberGenerator):
    """Implements a uniform real random number generator using tensorflow."""

    def __init__(self, seed: Optional[float] = None):
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
    def seed(self) -> Optional[float]:
        return self.__seed

    @seed.setter
    def seed(self, value: Optional[float]) -> None:
        self.__seed = value
        self.generator = get_rng(self.seed)
