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
        assert pytest.approx(result["log_likelihood"]) == -12961.752837852626
        assert pytest.approx(par_values["Width_f(0)(500)"]) == 0.5546203338476
        assert pytest.approx(par_errors["Width_f(0)(500)"]) == 0.00981961384643
        assert (
            pytest.approx(par_values["Position_f(0)(980)"])
            == 0.9897319832688249
        )
        assert (
            pytest.approx(par_errors["Position_f(0)(980)"])
            == 0.0007052210121507604
        )
