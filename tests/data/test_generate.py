"""Tests the generate facade functions."""

import pytest

from tensorwaves.data.generate import generate_data, generate_phsp

from tensorwaves.physics.helicityformalism.kinematics import HelicityKinematics


@pytest.mark.parametrize(
    "sample_size, kinematics_info, expected_shape",
    [
        (1000, (3.1, (0.13, 0.13, 0.13)), (3, 1000, 4)),
        (1000, (2.1, (0.13, 0.2, 0.1, 0.5)), (4, 1000, 4)),
        (5000, (3.1, (0.13, 0.13, 0.13, 0.2, 0.3)), (5, 5000, 4)),
    ],
)
def test_shape_generate_phsp(sample_size, kinematics_info, expected_shape):
    kin = HelicityKinematics(kinematics_info[0], kinematics_info[1])
    sample = generate_phsp(sample_size, kin)
    assert sample.shape == expected_shape
