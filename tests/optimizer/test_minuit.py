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
            "parameter_values",
            "parameter_errors",
            "log_likelihood",
            "function_calls",
            "time",
        }
        par_values = result["parameter_values"]
        assert set(par_values) == set(free_pars)
        assert pytest.approx(result["log_likelihood"]) == -13379.223862030514
        assert pytest.approx(par_values["Width_f(0)(500)"]) == 0.55868526502471
        assert pytest.approx(par_values["Mass_f(0)(980)"]) == 0.990141023090767
