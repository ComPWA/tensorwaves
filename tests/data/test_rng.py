# pylint:disable=import-outside-toplevel
import pytest

from tensorwaves.data.rng import (
    NumpyUniformRNG,
    TFUniformRealNumberGenerator,
    _get_tensorflow_rng,
)


class TestNumpyUniformRNG:
    def test_generate_deterministic(self):
        rng1 = NumpyUniformRNG(seed=0)
        rng2 = NumpyUniformRNG(seed=0)
        assert pytest.approx(rng1(size=3)) == [0.6369617, 0.2697867, 0.0409735]
        assert pytest.approx(rng2(size=2)) == [0.6369617, 0.2697867]
        assert pytest.approx(rng2(size=1)) == [0.0409735]

    def test_generate_indeterministic(self):
        rng1 = NumpyUniformRNG()
        rng2 = NumpyUniformRNG()
        with pytest.raises(AssertionError):
            assert pytest.approx(rng1(size=2)) == rng2(size=2)

    def test_reset_with_seed(self):
        rng = NumpyUniformRNG(seed=0)
        assert pytest.approx(rng(size=2)) == [0.6369617, 0.2697867]
        rng.seed = 0  # reset
        assert pytest.approx(rng(size=2)) == [0.6369617, 0.2697867]


class TestTFUniformRealNumberGenerator:
    @staticmethod
    def test_deterministic_call():
        generator = TFUniformRealNumberGenerator(seed=456)
        sample = generator(size=3, min_value=-1, max_value=+1)
        assert pytest.approx(sample) == [-0.38057342, -0.21197986, 0.14724727]


def test_get_tensorflow_rng():
    import tensorflow as tf

    for seed in [None, 100, tf.random.Generator.from_seed(seed=0)]:
        rng = _get_tensorflow_rng(seed)
        assert isinstance(rng, tf.random.Generator)
    with pytest.raises(
        TypeError, match=r"Cannot create a tf\.random\.Generator from a float"
    ):
        _get_tensorflow_rng(2.5)
