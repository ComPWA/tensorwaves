# pylint: disable=redefined-outer-name

import pytest


class TestMinuit2:
    @staticmethod
    def test_optimize(fit_result: dict, free_parameters: dict):
        result = fit_result
        assert set(result) == {
            "parameter_values",
            "parameter_errors",
            "log_likelihood",
            "function_calls",
            "execution_time",
        }
        par_values = result["parameter_values"]
        par_errors = result["parameter_errors"]
        assert set(par_values) == set(free_parameters)
        assert pytest.approx(result["log_likelihood"]) == -13379.223862030514
        assert pytest.approx(par_values["Width_f(0)(500)"]) == 0.55868526502471
        assert pytest.approx(par_errors["Width_f(0)(500)"]) == 0.01057804923356
        assert pytest.approx(par_values["Mass_f(0)(980)"]) == 0.990141023090767
        assert pytest.approx(par_errors["Mass_f(0)(980)"]) == 0.000721352674347
