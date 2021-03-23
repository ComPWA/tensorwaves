# pylint: disable=redefined-outer-name

import numpy as np
import pytest
import sympy as sp
from expertsystem.amplitude.helicity import HelicityModel

from tensorwaves.interfaces import DataSample, Function
from tensorwaves.model import (
    LambdifiedFunction,
    SympyModel,
    create_intensity_component,
)


@pytest.fixture(scope="module")
def function() -> LambdifiedFunction:
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
        + c_3 * (x ** 2 - 3 * x)
        + c_4
    )
    expression = sp.simplify((sp.conjugate(expression) * expression))
    model = SympyModel(expression=expression, parameters=params)
    return LambdifiedFunction(model, "numpy")


@pytest.mark.parametrize(
    "test_data, expected_results",
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


def test_create_intensity_component(
    phsp_set: DataSample,
    es_helicity_model: HelicityModel,
    intensity: LambdifiedFunction,
):
    # pylint: disable=line-too-long
    model = es_helicity_model
    from_amplitudes = create_intensity_component(
        model,
        components=[
            R"A[J/\psi(1S)_{+1} \to f_{0}(500)_{0} \gamma_{+1};f_{0}(500)_{0} \to \pi^{0}_{0} \pi^{0}_{0}]",
            R"A[J/\psi(1S)_{+1} \to f_{0}(980)_{0} \gamma_{+1};f_{0}(980)_{0} \to \pi^{0}_{0} \pi^{0}_{0}]",
        ],
        backend="numpy",
    )
    from_intensity = create_intensity_component(
        model,
        components=R"I[J/\psi(1S)_{+1} \to \gamma_{+1} \pi^{0}_{0} \pi^{0}_{0}]",
        backend="numpy",
    )
    assert pytest.approx(from_amplitudes(phsp_set)) == from_intensity(phsp_set)

    intensity_components = [
        create_intensity_component(model, component, backend="numpy")
        for component in model.components
        if component.startswith("I")
    ]
    sub_intensities = [i(phsp_set) for i in intensity_components]
    assert pytest.approx(sum(sub_intensities)) == intensity(phsp_set)
