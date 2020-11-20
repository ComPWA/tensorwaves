# pylint: disable=redefined-outer-name

import pytest

from tensorwaves.estimator import UnbinnedNLL
from tensorwaves.optimizer.minuit import Minuit2


class TestMinuit2:
    @staticmethod
    def test_optimize(estimator: UnbinnedNLL):
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
