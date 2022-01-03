# pylint: disable=import-outside-toplevel, no-self-use
import numpy as np
import pytest

from tensorwaves.data import (
    IdentityTransformer,
    IntensityDistributionGenerator,
    NumpyDomainGenerator,
    NumpyUniformRNG,
    TFPhaseSpaceGenerator,
    TFUniformRealNumberGenerator,
)
from tensorwaves.function.sympy import create_function
from tensorwaves.interface import DataSample, Function


class FlatDistribution(Function[DataSample, np.ndarray]):
    def __call__(self, data: DataSample) -> np.ndarray:
        some_key = next(iter(data))
        sample_size = len(data[some_key])
        return np.ones(sample_size)


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
            assert pytest.approx(array.min(), rel=1) == min_
            assert pytest.approx(array.max(), rel=0.1) == max_
            bin_content, _ = np.histogram(array, bins=10)
            bin_percentage = bin_content / np.size(array)
            assert pytest.approx(bin_percentage.std(), rel=1) == 0


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

    def test_generate_four_momenta_on_flat_distribution(self):
        sample_size = 5
        initial_state_mass = 3.0
        final_state_masses = {0: 0.135, 1: 0.135, 2: 0.135}
        phsp_generator = TFPhaseSpaceGenerator(
            initial_state_mass, final_state_masses
        )
        data_generator = IntensityDistributionGenerator(
            phsp_generator,
            function=FlatDistribution(),
            domain_transformer=IdentityTransformer(),
        )
        phsp = phsp_generator.generate(
            sample_size, rng=TFUniformRealNumberGenerator(seed=0)
        )
        data = data_generator.generate(
            sample_size, rng=TFUniformRealNumberGenerator(seed=0)
        )
        assert set(phsp) == {f"p{i}" for i in final_state_masses}
        assert set(phsp) == set(data)
        for i in phsp:
            assert pytest.approx(phsp[i]) == data[i]
