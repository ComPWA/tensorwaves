# pylint: disable=no-self-use
import pytest
from numpy.testing import assert_almost_equal

from tensorwaves.data import NumpyUniformRNG


class TestNumpyUniformRNG:
    def test_generate_deterministic(self):
        rng1 = NumpyUniformRNG(seed=0)
        rng2 = NumpyUniformRNG(seed=0)
        assert_almost_equal(rng1(size=3), [0.6369617, 0.2697867, 0.0409735])
        assert_almost_equal(rng2(size=2), [0.6369617, 0.2697867])
        assert_almost_equal(rng2(size=1), [0.0409735])

    def test_generate_indeterministic(self):
        rng1 = NumpyUniformRNG()
        rng2 = NumpyUniformRNG()
        with pytest.raises(AssertionError):
            assert_almost_equal(rng1(size=2), rng2(size=2))

    def test_reset_with_seed(self):
        rng = NumpyUniformRNG(seed=0)
        assert_almost_equal(rng(size=2), [0.6369617, 0.2697867])
        rng.seed = 0  # reset
        assert_almost_equal(rng(size=2), [0.6369617, 0.2697867])
