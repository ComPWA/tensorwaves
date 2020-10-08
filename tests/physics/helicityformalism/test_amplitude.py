import numpy as np
import pytest  # type: ignore

from tensorwaves.physics.helicity_formalism.amplitude import (
    _CoefficientAmplitude,
    _CoherentIntensity,
    _IncoherentIntensity,
    _SequentialAmplitude,
)


def linear_func(input_value):
    return input_value


@pytest.mark.parametrize(
    "functions, test_data, expected_results",
    [
        (
            [linear_func],
            [1.0, 2.0, 3.0, 4.2, 0.2],
            [1.0, 2.0, 3.0, 4.2, 0.2],
        ),
        (
            [linear_func, linear_func],
            [1.0, 2.0, 3.0, 4.2, 0.2],
            [2.0 * x for x in [1.0, 2.0, 3.0, 4.2, 0.2]],
        ),
    ],
)
def test_incoherent_intensity(functions, test_data, expected_results):
    model = _IncoherentIntensity(functions)
    results = model(test_data).numpy()
    np.testing.assert_array_almost_equal(results, expected_results, decimal=6)


@pytest.mark.parametrize(
    "functions, test_data, expected_results",
    [
        (
            [linear_func],
            [(1.0 + 2.0j), (1.5 - 1.4j), (0.12 + 20.0j)],
            [5.0, 4.21, 400.0144],
        ),
        (
            [linear_func, linear_func],
            [(1.0 + 2.0j), (1.5 - 1.4j), (-0.23 + 3.2j)],
            [20.0, 16.84, 41.1716],
        ),
    ],
)
def test_coherent_intensity(functions, test_data, expected_results):
    model = _CoherentIntensity(functions)
    results = model(test_data).numpy()
    np.testing.assert_array_almost_equal(results, expected_results, decimal=6)


@pytest.mark.parametrize(
    "function, mag, phase, test_data, expected_results",
    [
        (
            linear_func,
            2.0,
            0.0,
            [(1.0 + 2.0j), (1.5 - 1.4j), (0.12 + 20.0j)],
            [(2.0 + 4.0j), (3.0 - 2.8j), (0.24 + 40.0j)],
        ),
        (
            linear_func,
            3.0,
            0.5 * np.pi,
            [(1.0 + 2.0j), (1.5 - 1.4j), (-0.23 + 3.2j)],
            [(-6.0 + 3.0j), (4.2 + 4.5j), (-9.6 - 0.69j)],
        ),
    ],
)
def test_coefficient_amplitude(
    function, mag, phase, test_data, expected_results
):
    model = _CoefficientAmplitude(function, mag, phase)
    results = model(test_data).numpy()
    np.testing.assert_array_almost_equal(results, expected_results, decimal=6)


@pytest.mark.parametrize(
    "functions, test_data, expected_results",
    [
        (
            [linear_func],
            [(1.0 + 2.0j), (1.5 - 1.4j), (0.12 + 20.0j)],
            [(1.0 + 2.0j), (1.5 - 1.4j), (0.12 + 20.0j)],
        ),
        (
            [linear_func, linear_func],
            [(1.0 + 2.0j), (1.5 - 1.4j), (-0.23 + 3.2j)],
            [(-3.0 + 4.0j), (0.29 - 4.2j), (-10.1871 - 1.472j)],
        ),
    ],
)
def test_sequential_amplitude(functions, test_data, expected_results):
    model = _SequentialAmplitude(functions)
    results = model(test_data).numpy()
    np.testing.assert_array_almost_equal(results, expected_results, decimal=6)
