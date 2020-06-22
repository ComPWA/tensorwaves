"""Test helicity amplitude components."""

import math

import pytest  # type: ignore

from tensorwaves.physics.helicityformalism.amplitude import (
    _clebsch_gordan_coefficient,
    _determine_canonical_prefactor,
    _get_orbital_angular_momentum,
)


@pytest.mark.parametrize(
    "test_recipe, expected_value",
    [
        ({"J": 1.0, "M": 1.0, "j1": 0.5, "m1": 0.5, "j2": 0.5, "m2": 0.5}, 1,),
        (
            {"J": 1.0, "M": 0.0, "j1": 0.5, "m1": 0.5, "j2": 0.5, "m2": -0.5},
            math.sqrt(1 / 2),
        ),
        (
            {"J": 1.0, "M": 0.0, "j1": 0.5, "m1": -0.5, "j2": 0.5, "m2": 0.5},
            math.sqrt(1 / 2),
        ),
        (
            {"J": 0.0, "M": 0.0, "j1": 0.5, "m1": -0.5, "j2": 0.5, "m2": 0.5},
            -math.sqrt(1 / 2),
        ),
        (
            {"J": 0.0, "M": 0.0, "j1": 0.5, "m1": 0.5, "j2": 0.5, "m2": -0.5},
            math.sqrt(1 / 2),
        ),
        ({"J": 3.0, "M": 3.0, "j1": 2.0, "m1": 2.0, "j2": 1.0, "m2": 1.0}, 1,),
        (
            {"J": 3.0, "M": 2.0, "j1": 2.0, "m1": 2.0, "j2": 1.0, "m2": 0.0},
            math.sqrt(1 / 3),
        ),
        (
            {"J": 1.0, "M": 1.0, "j1": 2.0, "m1": 0.0, "j2": 1.0, "m2": 1.0},
            math.sqrt(1 / 10),
        ),
    ],
)
def test_clebsch_gordan_coefficent(test_recipe, expected_value):
    cgc = _clebsch_gordan_coefficient(test_recipe)
    assert cgc == pytest.approx(expected_value, rel=1e-6)


@pytest.mark.parametrize(
    "test_recipe, expected_value",
    [
        (
            {
                "LS": {
                    "ClebschGordan": {
                        "J": 1.5,
                        "M": 0.5,
                        "j1": 1.0,
                        "m1": 0.0,
                        "j2": 0.5,
                        "m2": 0.5,
                    }
                }
            },
            1.0,
        ),
    ],
)
def test_orbital_angular_momentum(test_recipe, expected_value):
    orbit_l = _get_orbital_angular_momentum(test_recipe)
    assert orbit_l == pytest.approx(expected_value, rel=1e-6)


@pytest.mark.parametrize(
    "test_recipe, expected_value",
    [
        (
            {
                "LS": {
                    "ClebschGordan": {
                        "J": 1.0,
                        "M": 1.0,
                        "j1": 2.0,
                        "m1": 0.0,
                        "j2": 1.0,
                        "m2": 1.0,
                    }
                },
                "s2s3": {
                    "ClebschGordan": {
                        "J": 1.0,
                        "M": 1.0,
                        "j1": 0.0,
                        "m1": 0.0,
                        "j2": 1.0,
                        "m2": 1.0,
                    }
                },
            },
            math.sqrt(1 / 10) * 1,
        ),
        (
            {
                "LS": {
                    "ClebschGordan": {
                        "J": 1.0,
                        "M": 1.0,
                        "j1": 2.0,
                        "m1": 0.0,
                        "j2": 1.0,
                        "m2": 1.0,
                    }
                },
                "s2s3": {
                    "ClebschGordan": {
                        "J": 1.0,
                        "M": 1.0,
                        "j1": 1.0,
                        "m1": 0.0,
                        "j2": 1.0,
                        "m2": 1.0,
                    }
                },
            },
            math.sqrt(1 / 10) * -math.sqrt(1 / 2),
        ),
        (
            {
                "LS": {
                    "ClebschGordan": {
                        "J": 1.0,
                        "M": 1.0,
                        "j1": 2.0,
                        "m1": 0.0,
                        "j2": 1.0,
                        "m2": 1.0,
                    }
                },
            },
            KeyError(),
        ),
        (
            {
                "LS": {
                    "ClebschGordan": {
                        "J": 1.0,
                        "M": 1.0,
                        "j1": 2.0,
                        "m1": 0.0,
                        "j2": 1.0,
                        "m2": 1.0,
                    }
                },
                "s2s3": {
                    "ClebschGordan": {
                        "J": 1.0,
                        "m": 1.0,
                        "j1": 1.0,
                        "m1": 0.0,
                        "j2": 1.0,
                        "m2": 1.0,
                    }
                },
            },
            KeyError(),
        ),
    ],
)
def test_determine_canonical_prefactor(test_recipe, expected_value):
    if isinstance(expected_value, BaseException):
        with pytest.raises(type(expected_value)):
            prefactor = _determine_canonical_prefactor(test_recipe)
    else:
        prefactor = _determine_canonical_prefactor(test_recipe)
        assert prefactor == pytest.approx(expected_value, rel=1e-6)
