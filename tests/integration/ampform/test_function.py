# pylint: disable=line-too-long, no-self-use, redefined-outer-name
import qrules

from tensorwaves.function import SympyModel


class TestSympyModel:
    def test_parameters(
        self, reaction: qrules.ReactionInfo, sympy_model: SympyModel
    ):
        if reaction.formalism == "canonical-helicity":
            assert set(sympy_model.parameters) == {
                R"C_{J/\psi(1S) \xrightarrow[S=1]{L=0} f_{0}(500) \gamma;"
                R" f_{0}(500)"
                R" \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}",
                R"C_{J/\psi(1S) \xrightarrow[S=1]{L=0} f_{0}(980) \gamma;"
                R" f_{0}(980)"
                R" \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}",
                R"C_{J/\psi(1S) \xrightarrow[S=1]{L=2} f_{0}(500) \gamma;"
                R" f_{0}(500)"
                R" \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}",
                R"C_{J/\psi(1S) \xrightarrow[S=1]{L=2} f_{0}(980) \gamma;"
                R" f_{0}(980)"
                R" \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}",
                "Gamma_f(0)(500)",
                "Gamma_f(0)(980)",
                "m_f(0)(500)",
                "m_f(0)(980)",
            }
        else:
            assert set(sympy_model.parameters) == {
                R"C_{J/\psi(1S) \to f_{0}(980)_{0} \gamma_{+1}; f_{0}(980) \to"
                R" \pi^{0}_{0} \pi^{0}_{0}}",
                R"C_{J/\psi(1S) \to f_{0}(500)_{0} \gamma_{+1}; f_{0}(500) \to"
                R" \pi^{0}_{0} \pi^{0}_{0}}",
                "m_f(0)(980)",
                "Gamma_f(0)(980)",
                "m_f(0)(500)",
                "Gamma_f(0)(500)",
            }
