import numpy as np
import pytest

from tensorwaves.data.rng import (
    JAXUniformRNG,
    NumpyUniformRNG,
    TFUniformRealNumberGenerator,
    UniformRNG,
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


@pytest.mark.tensorflow
def describe_TFUniformRealNumberGenerator():
    def it_generates_the_same_sequence_for_the_same_seed():
        generator = TFUniformRealNumberGenerator(seed=456)
        sample = generator(size=3, min_value=-1, max_value=+1)
        assert pytest.approx(sample) == [-0.38057342, -0.21197986, 0.14724727]


@pytest.mark.tensorflow
def describe_get_tensorflow_rng():
    def it_accepts_a_seed_a_generator_or_nothing():
        import tensorflow as tf

        for seed in [None, 100, tf.random.Generator.from_seed(seed=0)]:
            rng = _get_tensorflow_rng(seed)
            assert isinstance(rng, tf.random.Generator)


def describe_JAXUniformRNG():
    def it_generates_the_same_sequence_for_the_same_seed():
        rng1 = JAXUniformRNG(seed=0)
        rng2 = JAXUniformRNG(seed=0)
        values = rng1(size=3)
        assert values.shape == (3,)
        np.testing.assert_array_equal(rng2(size=3), values)
        assert not np.array_equal(rng2(size=3), values)

    def it_restarts_the_sequence_when_the_seed_is_set_again():
        rng = JAXUniformRNG(seed=0)
        values = rng(size=2)
        rng.seed = 0
        np.testing.assert_array_equal(rng(size=2), values)

    def it_respects_the_range():
        values = JAXUniformRNG(seed=0)(size=100, min_value=2.0, max_value=3.0)
        assert np.all((values >= 2.0) & (values < 3.0))


def describe_UniformRNG():
    @pytest.mark.parametrize(
        ("backend", "expected_type"),
        [
            ("numpy", NumpyUniformRNG),
            ("jax", JAXUniformRNG),
            ("tensorflow", TFUniformRealNumberGenerator),
        ],
    )
    def it_delegates_to_the_backend_implementation(backend, expected_type):
        rng = UniformRNG(seed=0, backend=backend)
        assert rng.backend == backend
        assert type(rng.implementation) is expected_type
        reference = expected_type(seed=0)
        np.testing.assert_array_equal(rng(size=3), reference(size=3))
        rng.seed = 1
        assert rng.implementation.seed == 1

    def it_rejects_an_unknown_backend():
        with pytest.raises(ValueError, match="Backend must be one of"):
            UniformRNG(backend="torch")  # ty: ignore[invalid-argument-type]
