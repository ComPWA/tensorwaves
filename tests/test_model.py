# pylint: disable=line-too-long, redefined-outer-name
import numpy as np
import pytest
import sympy as sp

from tensorwaves.interfaces import DataSample, Function
from tensorwaves.model import LambdifiedFunction, SympyModel


@pytest.fixture(scope="module")
def sympy_model() -> SympyModel:
    c_1, c_2, c_3, c_4 = sp.symbols("c_(1:5)")
    x = sp.Symbol("x", real=True)
    params = {
        c_1: 1 + 1j,
        c_2: -1 + 1j,
        c_3: 1 - 1j,
        c_4: -1 - 1j,
    }
    expression = (
        c_1 * sp.sqrt(x) / x
        + c_2 * sp.exp(-sp.Rational(1, 2) * ((x - 2) / sp.Rational(1, 2)) ** 2)
        + c_3 * (x**2 - 3 * x)
        + c_4
    )
    expression = sp.simplify(sp.conjugate(expression) * expression)
    return SympyModel(expression=expression, parameters=params)


@pytest.fixture(scope="module")
def function(sympy_model) -> LambdifiedFunction:
    return LambdifiedFunction(sympy_model, "numpy")


@pytest.mark.parametrize(
    ("test_data", "expected_results"),
    [
        (
            {"x": np.array([0.5, 1.0, 1.5, 2.0, 2.5])},
            [3.52394, 9.11931, 16.3869, 18.1716, 7.16359],
        ),
    ],
)
def test_complex_amplitude(
    function: Function,
    test_data: DataSample,
    expected_results: np.ndarray,
):
    results = function(test_data)
    np.testing.assert_array_almost_equal(results, expected_results, decimal=4)


def test_canonical(canonical_model: SympyModel):
    assert set(canonical_model.parameters) == {
        R"C[J/\psi(1S) \to f_{0}(980)_{0} \gamma_{+1}; f_{0}(980) \to"
        R" \pi^{0}_{0} \pi^{0}_{0}]",
        R"C[J/\psi(1S) \to f_{0}(500)_{0} \gamma_{+1}; f_{0}(500) \to"
        R" \pi^{0}_{0} \pi^{0}_{0}]",
        "m_f(0)(980)",
        "Gamma_f(0)(980)",
        "m_f(0)(500)",
        "Gamma_f(0)(500)",
    }


def test_helicity(helicity_model: SympyModel):
    assert set(helicity_model.parameters) == {
        R"C[J/\psi(1S) \to f_{0}(980)_{0} \gamma_{+1}; f_{0}(980) \to"
        R" \pi^{0}_{0} \pi^{0}_{0}]",
        R"C[J/\psi(1S) \to f_{0}(500)_{0} \gamma_{+1}; f_{0}(500) \to"
        R" \pi^{0}_{0} \pi^{0}_{0}]",
        "m_f(0)(980)",
        "Gamma_f(0)(980)",
        "m_f(0)(500)",
        "Gamma_f(0)(500)",
    }
