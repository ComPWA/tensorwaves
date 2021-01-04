from tensorwaves.estimator import UnbinnedNLL


class TestUnbinnedNLL:
    @staticmethod
    def test_parameters(estimator: UnbinnedNLL):
        assert estimator.parameters == {
            "strength_incoherent": 1.0,
            "MesonRadius_J/psi(1S)": 1.0,
            "MesonRadius_f(0)(500)": 1.0,
            "MesonRadius_f(0)(980)": 1.0,
            "Magnitude_J/psi(1S)_to_f(0)(500)_0+gamma_1;f(0)(500)_to_pi0_0+pi0_0;": 1.0,
            "Phase_J/psi(1S)_to_f(0)(500)_0+gamma_1;f(0)(500)_to_pi0_0+pi0_0;": 0.0,
            "Magnitude_J/psi(1S)_to_f(0)(980)_0+gamma_1;f(0)(980)_to_pi0_0+pi0_0;": 1.0,
            "Phase_J/psi(1S)_to_f(0)(980)_0+gamma_1;f(0)(980)_to_pi0_0+pi0_0;": 0.0,
            "Position_J/psi(1S)": 3.0969,
            "Width_J/psi(1S)": 9.29e-05,
            "Position_f(0)(500)": 0.475,
            "Width_f(0)(500)": 0.55,
            "Position_f(0)(980)": 0.99,
            "Width_f(0)(980)": 0.06,
        }
