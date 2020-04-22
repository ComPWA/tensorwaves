from tensorwaves.interfaces import (
    PhaseSpaceGenerator, UniformRealNumberGenerator)

import numpy as np
import phasespace
import tensorflow as tf


class TFPhaseSpaceGenerator(PhaseSpaceGenerator):
    def __init__(self, initial_state_mass, final_state_masses):
        print(phasespace.__file__)
        self.phsp_gen = phasespace.nbody_decay(initial_state_mass,
                                               final_state_masses)

    def generate(self, size, random_generator):
        # TODO: phasespace has to be improved to accept a random generator
        # to ensure deterministic behavior based on the seed
        w, p = self.phsp_gen.generate(n_events=size)
        p = np.array(tuple(p[x].numpy() for x in p.keys()))
        return p, w


class TFUniformRealNumberGenerator(UniformRealNumberGenerator):
    def __init__(self, seed):
        self.seed = seed
        self.random = tf.random.uniform
        self.dtype = tf.float64

    def __call__(self, size, min_value=0.0, max_value=1.0):
        return self.random(shape=[size, ], minval=min_value,
                           maxval=max_value, dtype=self.dtype)

    @property
    def seed(self):
        return self.__seed

    @seed.setter
    def seed(self, value: float):
        self.__seed = value
        tf.random.set_seed(self.__seed)
