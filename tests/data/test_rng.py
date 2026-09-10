import pytest

from tensorwaves.data.rng import (
    NumpyUniformRNG,
    TFUniformRealNumberGenerator,
    _get_tensorflow_rng,
)


def describe_NumpyUniformRNG():
    def it_generates_the_same_sequence_for_the_same_seed():
        rng1 = NumpyUniformRNG(seed=0)
        rng2 = NumpyUniformRNG(seed=0)
        assert pytest.approx(rng1(size=3)) == [0.6369617, 0.2697867, 0.0409735]
        assert pytest.approx(rng2(size=2)) == [0.6369617, 0.2697867]
        assert pytest.approx(rng2(size=1)) == [0.0409735]

    def it_generates_different_sequences_without_a_seed():
        rng1 = NumpyUniformRNG()
        rng2 = NumpyUniformRNG()
        with pytest.raises(AssertionError):
            assert pytest.approx(rng1(size=2)) == rng2(size=2)

    def it_restarts_the_sequence_when_the_seed_is_set_again():
        rng = NumpyUniformRNG(seed=0)
        assert pytest.approx(rng(size=2)) == [0.6369617, 0.2697867]
        rng.seed = 0  # reset
        assert pytest.approx(rng(size=2)) == [0.6369617, 0.2697867]


def describe_TFUniformRealNumberGenerator():
    def it_generates_the_same_sequence_for_the_same_seed():
        generator = TFUniformRealNumberGenerator(seed=456)
        sample = generator(size=3, min_value=-1, max_value=+1)
        assert pytest.approx(sample) == [-0.38057342, -0.21197986, 0.14724727]


def describe_get_tensorflow_rng():
    def it_accepts_a_seed_a_generator_or_nothing():
        import tensorflow as tf

        for seed in [None, 100, tf.random.Generator.from_seed(seed=0)]:
            rng = _get_tensorflow_rng(seed)
            assert isinstance(rng, tf.random.Generator)
