# pylint:disable=import-outside-toplevel
"""Implementations of `.RealNumberGenerator`."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from tensorwaves.function._backend import raise_missing_module_error
from tensorwaves.interface import RealNumberGenerator

if TYPE_CHECKING:  # pragma: no cover
    import tensorflow as tf

    SeedLike = Optional[Union[int, tf.random.Generator]]


class NumpyUniformRNG(RealNumberGenerator):
    """Implements a uniform real random number generator using `numpy`."""

    def __init__(self, seed: float | None = None):
        self.seed = seed

    def __call__(
        self, size: int, min_value: float = 0.0, max_value: float = 1.0
    ) -> np.ndarray:
        return self.generator.uniform(size=size, low=min_value, high=max_value)

    @property
    def seed(self) -> float | None:
        return self.__seed

    @seed.setter
    def seed(self, value: float | None) -> None:
        self.__seed = value
        generator_seed: float | int | None = self.seed
        if generator_seed is not None:
            if not float(generator_seed).is_integer():
                raise ValueError("NumPy generator seed has to be integer")
            generator_seed = int(generator_seed)
        self.generator: np.random.Generator = np.random.default_rng(
            seed=generator_seed
        )


class TFUniformRealNumberGenerator(RealNumberGenerator):
    """Implements a uniform real random number generator using tensorflow."""

    def __init__(self, seed: float | None = None):
        try:
            from tensorflow import float64
        except ImportError:  # pragma: no cover
            raise_missing_module_error("tensorflow", extras_require="tf")

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
    def seed(self) -> float | None:
        return self.__seed

    @seed.setter
    def seed(self, value: float | None) -> None:
        self.__seed = value
        self.generator = _get_tensorflow_rng(self.seed)


def _get_tensorflow_rng(seed: SeedLike = None) -> tf.random.Generator:
    """Get or create a `tf.random.Generator`.

    https://github.com/zfit/phasespace/blob/5998e2b/phasespace/random.py#L15-L41
    """
    try:
        import tensorflow as tf
    except ImportError:  # pragma: no cover
        raise_missing_module_error("tensorflow", extras_require="tf")

    if seed is None:
        return tf.random.get_global_generator()
    if isinstance(seed, int):
        return tf.random.Generator.from_seed(seed=seed)
    if isinstance(seed, tf.random.Generator):
        return seed
    raise TypeError(
        f"Cannot create a tf.random.Generator from a {type(seed).__name__}"
    )
