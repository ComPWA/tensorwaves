# pylint: disable=redefined-outer-name

import numpy as np
import pytest
import sympy as sy

from tensorwaves.physics.amplitude import SympyModel


@pytest.fixture(scope="module")
def function() -> SympyModel:
    c_1, c_2, c_3, c_4 = sy.symbols("c_1,c_2,c_3,c_4")
    x = sy.Symbol("x", real=True)
    params = {
        c_1: 1 + 1j,
        c_2: -1 + 1j,
        c_3: 1 - 1j,
        c_4: -1 - 1j,
    }
    sympy_expr = (
        c_1 * sy.sqrt(x) / x
        + c_2 * sy.exp(-sy.Rational(1, 2) * ((x - 2) / sy.Rational(1, 2)) ** 2)
        + c_3 * (x ** 2 - 3 * x)
        + c_4
    )
    sympy_expr = sympy_expr.subs(params)
    sympy_expr = sy.simplify((sy.conjugate(sympy_expr) * sympy_expr))
    return SympyModel(expression=sympy_expr, parameters=params)


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


def test_helicity(helicity_model: SympyModel):
    assert set(helicity_model.parameters) == {
        "C[J/\\psi(1S) \\to f_{0}(980)_{0} \\gamma_{+1};f_{0}(980) \\to \\pi^{0}_{0} \\pi^{0}_{0}]",
        "C[J/\\psi(1S) \\to f_{0}(500)_{0} \\gamma_{+1};f_{0}(500) \\to \\pi^{0}_{0} \\pi^{0}_{0}]",
        "m_f(0)(980)",
        "d_f(0)(980)",
        "Gamma_f(0)(980)",
        "m_f(0)(500)",
        "d_f(0)(500)",
        "Gamma_f(0)(500)",
    }


def test_canonical(canonical_model: SympyModel):
    assert set(canonical_model.parameters) == {
        "C[J/\\psi(1S) \\to f_{0}(980)_{0} \\gamma_{+1};f_{0}(980) \\to \\pi^{0}_{0} \\pi^{0}_{0}]",
        "C[J/\\psi(1S) \\to f_{0}(500)_{0} \\gamma_{+1};f_{0}(500) \\to \\pi^{0}_{0} \\pi^{0}_{0}]",
        "m_f(0)(980)",
        "d_f(0)(980)",
        "Gamma_f(0)(980)",
        "m_f(0)(500)",
        "d_f(0)(500)",
        "Gamma_f(0)(500)",
    }
