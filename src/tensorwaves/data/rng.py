# pylint:disable=import-outside-toplevel
"""Implementations of `.RealNumberGenerator`."""

from typing import Optional

import numpy as np

from tensorwaves.interface import RealNumberGenerator


class NumpyUniformRNG(RealNumberGenerator):
    """Implements a uniform real random number generator using `numpy`."""

    def __init__(self, seed: Optional[float] = None):
        self.seed = seed

    def __call__(
        self, size: int, min_value: float = 0.0, max_value: float = 1.0
    ) -> np.ndarray:
        return self.generator.uniform(size=size, low=min_value, high=max_value)

    @property
    def seed(self) -> Optional[float]:
        return self.__seed

    @seed.setter
    def seed(self, value: Optional[float]) -> None:
        self.__seed = value
        self.generator: np.random.Generator = np.random.default_rng(
            seed=self.seed
        )


class TFUniformRealNumberGenerator(RealNumberGenerator):
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
