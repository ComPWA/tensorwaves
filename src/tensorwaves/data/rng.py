# pylint:disable=import-outside-toplevel
"""Implementations of `.RealNumberGenerator`."""

from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from tensorwaves.interface import RealNumberGenerator

if TYPE_CHECKING:  # pragma: no cover
    import tensorflow as tf

    SeedLike = Optional[Union[int, tf.random.Generator]]


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
        self.__seed = value
        self.generator = _get_tensorflow_rng(self.seed)


def _get_tensorflow_rng(seed: "SeedLike" = None) -> "tf.random.Generator":
    """Get or create a `tf.random.Generator`.

    https://github.com/zfit/phasespace/blob/5998e2b/phasespace/random.py#L15-L41
    """
    import tensorflow as tf

    if seed is None:
        return tf.random.get_global_generator()
    if isinstance(seed, int):
        return tf.random.Generator.from_seed(seed=seed)
    if isinstance(seed, tf.random.Generator):
        return seed
    raise TypeError(
        f"Cannot create a tf.random.Generator from a {type(seed).__name__}"
    )
