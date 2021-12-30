# pylint: disable=import-outside-toplevel, no-self-use
import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from tensorwaves.data import (
    IntensityDistributionGenerator,
    NumpyDomainGenerator,
    NumpyUniformRNG,
)
from tensorwaves.function.sympy import create_function


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


class TestNumpyDomainGenerator:
    def test_generate(self):
        rng = NumpyUniformRNG(seed=0)
        boundaries = {
            "x": (0.0, 5.0),
            "y": (2.0, 3.0),
            "z": (-1.0, 1.0),
        }
        domain_generator = NumpyDomainGenerator(boundaries)
        domain_sample = domain_generator.generate(10_000, rng)
        assert set(domain_sample) == set(boundaries)
        for variable_name, array in domain_sample.items():
            min_, max_ = boundaries[variable_name]
            assert_almost_equal(array.min(), min_, decimal=3)
            assert_almost_equal(array.max(), max_, decimal=3)
            bin_content, _ = np.histogram(array, bins=10)
            bin_percentage = bin_content / np.size(array)
            assert_almost_equal(bin_percentage.std(), 0, decimal=2)


class TestIntensityDistributionGenerator:
    def test_generate(self):
        import sympy as sp

        x = sp.Symbol("x")
        heaviside_expr = sp.Piecewise((0, x < 0), (1, True))
        heaviside_func = create_function(heaviside_expr, backend="numpy")
        rng = NumpyUniformRNG(seed=0)
        domain_generator = NumpyDomainGenerator(boundaries={"x": (-1.0, +1.0)})
        data_generator = IntensityDistributionGenerator(
            domain_generator, heaviside_func, bunch_size=1_000
        )
        size = 10_000
        data = data_generator.generate(size, rng)
        assert set(data) == {"x"}
        x_data = data["x"]
        assert len(x_data) == size
        assert len(x_data[x_data < 0]) == 0
        assert len(x_data[x_data >= 0]) == size
        assert (
            pytest.approx(len(x_data[x_data >= 0.5]) / size, abs=0.01) == 0.5
        )
