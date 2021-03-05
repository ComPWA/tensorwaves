# pylint: disable=redefined-outer-name

import numpy as np
import pytest
import sympy as sp
from expertsystem.amplitude.helicity import HelicityModel

from tensorwaves.physics.amplitude import Intensity


@pytest.fixture(scope="module")
def function() -> Intensity:
    c_1, c_2, c_3, c_4 = sp.symbols("c_1,c_2,c_3,c_4")
    x = sp.Symbol("x", real=True)
    parameters = {
        c_1: 1 + 1j,
        c_2: -1 + 1j,
        c_3: 1 - 1j,
        c_4: -1 - 1j,
    }
    expression = (
        c_1 * sp.sqrt(x) / x
        + c_2 * sp.exp(-sp.Rational(1, 2) * ((x - 2) / sp.Rational(1, 2)) ** 2)
        + c_3 * (x ** 2 - 3 * x)
        + c_4
    )
    expression = expression.subs(parameters)
    expression = sp.simplify((sp.conjugate(expression) * expression))
    return Intensity(expression, parameters)


@pytest.mark.parametrize(
    "test_data, expected_results",
    [
        (
            {"x": np.array([0.5, 1.0, 1.5, 2.0, 2.5])},
            [3.52394, 9.11931, 16.3869, 18.1716, 7.16359],
        ),
    ],
)
def test_complex_amplitude(function, test_data, expected_results):
    results = function(test_data)
    np.testing.assert_array_almost_equal(results, expected_results, decimal=4)


def test_helicity(helicity_model: HelicityModel):
    intensity = Intensity(helicity_model.expression, helicity_model.parameters)

    assert set(intensity.parameters) == {
        "C[J/\\psi(1S) \\to f_{0}(980)_{0} \\gamma_{+1};f_{0}(980) \\to \\pi^{0}_{0} \\pi^{0}_{0}]",
        "C[J/\\psi(1S) \\to f_{0}(500)_{0} \\gamma_{+1};f_{0}(500) \\to \\pi^{0}_{0} \\pi^{0}_{0}]",
        "m_f(0)(980)",
        "d_f(0)(980)",
        "Gamma_f(0)(980)",
        "m_f(0)(500)",
        "d_f(0)(500)",
        "Gamma_f(0)(500)",
    }


def test_canonical(canonical_model: HelicityModel):
    intensity = Intensity(
        canonical_model.expression, canonical_model.parameters
    )
    assert set(intensity.parameters) == {
        "C[J/\\psi(1S) \\to f_{0}(980)_{0} \\gamma_{+1};f_{0}(980) \\to \\pi^{0}_{0} \\pi^{0}_{0}]",
        "C[J/\\psi(1S) \\to f_{0}(500)_{0} \\gamma_{+1};f_{0}(500) \\to \\pi^{0}_{0} \\pi^{0}_{0}]",
        "m_f(0)(980)",
        "d_f(0)(980)",
        "Gamma_f(0)(980)",
        "m_f(0)(500)",
        "d_f(0)(500)",
        "Gamma_f(0)(500)",
    }
