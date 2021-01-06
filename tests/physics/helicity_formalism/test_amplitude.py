import expertsystem.amplitude.model as es
import numpy as np
import pytest  # type: ignore
import tensorflow as tf

from tensorwaves.physics.helicity_formalism.amplitude import (
    IntensityBuilder,
    _CoefficientAmplitude,
    _CoherentIntensity,
    _create_dynamics,
    _IncoherentIntensity,
    _SequentialAmplitude,
)
from tensorwaves.physics.helicity_formalism.kinematics import (
    HelicityKinematics,
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
    model = _CoefficientAmplitude(
        function,
        tf.constant(mag, dtype=tf.float64),
        tf.constant(phase, dtype=tf.float64),
    )
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


@pytest.mark.parametrize(
    "decaying_particle_name, valid",
    [
        ("p", False),
        ("pi0", True),
    ],
)
def test_invalid_angular_momentum_error(decaying_particle_name, valid, pdg):
    kinematics = HelicityKinematics(None)  # type: ignore
    builder = IntensityBuilder(pdg, kinematics)
    # pylint: disable=protected-access
    builder._dynamics = es.ParticleDynamics(pdg, parameters=es.FitParameters())
    builder._dynamics.set_breit_wigner(decaying_particle_name)
    dec_prod_fs_ids = [[0], [1]]
    decaying_particle = es.HelicityParticle(
        particle=pdg.find(decaying_particle_name), helicity=0
    )
    inv_mass_name = "foo"

    amplitude_node = es.HelicityDecay(
        decaying_particle=decaying_particle, decay_products=[]
    )

    if not valid:
        with pytest.raises(
            ValueError, match=r".*Model invalid.*angular momentum.*"
        ):
            _create_dynamics(
                builder=builder,
                amplitude_node=amplitude_node,
                dec_prod_fs_ids=dec_prod_fs_ids,
                decaying_state=decaying_particle,
                inv_mass_name=inv_mass_name,
                kinematics=kinematics,
            )
    else:
        _create_dynamics(
            builder=builder,
            amplitude_node=amplitude_node,
            dec_prod_fs_ids=dec_prod_fs_ids,
            decaying_state=decaying_particle,
            inv_mass_name=inv_mass_name,
            kinematics=kinematics,
        )
