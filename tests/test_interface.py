import math

import pytest
from IPython.lib.pretty import pretty

from tensorwaves.interface import FitResult


def describe_FitResult():
    @pytest.fixture(scope="module")
    def fit_result() -> FitResult:
        return FitResult(
            minimum_valid=True,
            execution_time=1.0,
            function_calls=50,
            estimator_value=-2e4,
            parameter_values={
                R"\pi": math.pi,
                "a": -5.0,
                "complex": 1 + 1j,
            },
        )

    def it_counts_a_complex_parameter_as_one_or_two(fit_result: FitResult):
        assert fit_result.count_number_of_parameters(complex_twice=False) == 3
        assert fit_result.count_number_of_parameters(complex_twice=True) == 4

    def it_has_an_evaluatable_pretty_repr(fit_result: FitResult):
        src = pretty(fit_result)
        reconstructed = eval(src)
        assert fit_result == reconstructed
