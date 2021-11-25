# pylint: disable=redefined-outer-name
from tensorwaves.model import SympyModel


def test_canonical(canonical_model: SympyModel):
    assert set(canonical_model.parameters) == {
        R"C_{J/\psi(1S) \xrightarrow[S=1]{L=0} f_{0}(500) \gamma; f_{0}(500)"
        R" \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}",
        R"C_{J/\psi(1S) \xrightarrow[S=1]{L=0} f_{0}(980) \gamma; f_{0}(980)"
        R" \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}",
        R"C_{J/\psi(1S) \xrightarrow[S=1]{L=2} f_{0}(500) \gamma; f_{0}(500)"
        R" \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}",
        R"C_{J/\psi(1S) \xrightarrow[S=1]{L=2} f_{0}(980) \gamma; f_{0}(980)"
        R" \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}",
        "Gamma_f(0)(500)",
        "Gamma_f(0)(980)",
        "m_f(0)(500)",
        "m_f(0)(980)",
    }


def test_helicity(helicity_model: SympyModel):
    assert set(helicity_model.parameters) == {
        R"C_{J/\psi(1S) \to f_{0}(980)_{0} \gamma_{+1}; f_{0}(980) \to"
        R" \pi^{0}_{0} \pi^{0}_{0}}",
        R"C_{J/\psi(1S) \to f_{0}(500)_{0} \gamma_{+1}; f_{0}(500) \to"
        R" \pi^{0}_{0} \pi^{0}_{0}}",
        "m_f(0)(980)",
        "Gamma_f(0)(980)",
        "m_f(0)(500)",
        "Gamma_f(0)(500)",
    }
