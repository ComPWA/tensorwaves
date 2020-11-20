# pylint: disable=redefined-outer-name

import pytest

from tensorwaves.estimator import UnbinnedNLL
from tensorwaves.optimizer.minuit import Minuit2
from tensorwaves.physics.helicity_formalism.amplitude import IntensityTF


class TestMinuit2:
    @staticmethod
    def test_optimize(intensity: IntensityTF, data_set: dict):
        estimator = UnbinnedNLL(intensity, data_set)
        assert estimator.parameters == {
            "strength_incoherent": 1.0,
            "MesonRadius_J/psi(1S)": 1.0,
            "MesonRadius_f(0)(500)": 1.0,
            "MesonRadius_f(0)(980)": 1.0,
            "Magnitude_J/psi(1S)_to_f(0)(500)_0+gamma_1;f(0)(500)_to_pi0_0+pi0_0;": 1.0,
            "Phase_J/psi(1S)_to_f(0)(500)_0+gamma_1;f(0)(500)_to_pi0_0+pi0_0;": 0.0,
            "Magnitude_J/psi(1S)_to_f(0)(980)_0+gamma_1;f(0)(980)_to_pi0_0+pi0_0;": 1.0,
            "Phase_J/psi(1S)_to_f(0)(980)_0+gamma_1;f(0)(980)_to_pi0_0+pi0_0;": 0.0,
            "Mass_J/psi(1S)": 3.0969,
            "Width_J/psi(1S)": 9.29e-05,
            "Mass_f(0)(500)": 0.475,
            "Width_f(0)(500)": 0.55,
            "Mass_f(0)(980)": 0.99,
            "Width_f(0)(980)": 0.06,
        }
        free_pars = {
            "Width_f(0)(500)": 0.3,
            "Mass_f(0)(980)": 1,
        }
        optimizer = Minuit2()
        result = optimizer.optimize(estimator, free_pars)
        assert set(result) == {
            "params",
            "log_lh",
            "func_calls",
            "time",
        }
        assert set(result["params"]) == set(free_pars)
        assert pytest.approx(result["log_lh"]) == -13379.223862030514
        assert pytest.approx(free_pars["Width_f(0)(500)"]) == 0.559522579972911
        assert pytest.approx(free_pars["Mass_f(0)(980)"]) == 0.9901984320598398
