"""Phase space generation using tensorflow."""

from typing import Optional, Tuple

import expertsystem.amplitude.kinematics as es
import phasespace
import tensorflow as tf
from expertsystem.amplitude.data import MomentumPool, ScalarSequence
from phasespace.random import get_rng

from tensorwaves.interfaces import (
    PhaseSpaceGenerator,
    UniformRealNumberGenerator,
)


class TFPhaseSpaceGenerator(PhaseSpaceGenerator):
    """Implements a phase space generator using tensorflow."""

    def __init__(self, reaction_info: es.ReactionInfo) -> None:
        initial_states = reaction_info.initial_state.values()
        if len(initial_states) != 1:
            raise ValueError("Not a 1-to-n body decay")
        initial_state = next(iter(initial_states))
        self.phsp_gen = phasespace.nbody_decay(
            mass_top=initial_state.mass,
            masses=[p.mass for p in reaction_info.final_state.values()],
            names=list(map(str, reaction_info.final_state)),
        )

    def generate(
        self, size: int, rng: UniformRealNumberGenerator
    ) -> Tuple[MomentumPool, ScalarSequence]:
        if not isinstance(rng, TFUniformRealNumberGenerator):
            raise TypeError(
                f"{TFPhaseSpaceGenerator.__name__} requires a "
                f"{TFUniformRealNumberGenerator.__name__}, but fed a "
                f"{rng.__class__.__name__}"
            )
        weights, particles = self.phsp_gen.generate(
            n_events=size, seed=rng.generator
        )
        momentum_pool = MomentumPool(
            {
                int(label): momenta.numpy()[:, [3, 0, 1, 2]]
                for label, momenta in particles.items()
            }
        )
        return momentum_pool, ScalarSequence(weights.numpy())


class TFUniformRealNumberGenerator(UniformRealNumberGenerator):
    """Implements a uniform real random number generator using tensorflow."""

    def __init__(self, seed: Optional[float] = None):
        self.seed = seed
        self.dtype = tf.float64

    def __call__(
        self, size: int, min_value: float = 0.0, max_value: float = 1.0
    ) -> ScalarSequence:
        return ScalarSequence(
            self.generator.uniform(
                shape=[size],
                minval=min_value,
                maxval=max_value,
                dtype=self.dtype,
            ).numpy()
        )

    @property
    def seed(self) -> Optional[float]:
        return self.__seed

    @seed.setter
    def seed(self, value: Optional[float]) -> None:
        self.__seed = value
        self.generator = get_rng(self.seed)
