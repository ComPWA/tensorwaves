# pylint: disable=eval-used
import pytest
from IPython.lib.pretty import pretty

from tensorwaves.interface import FitResult


class TestFitResult:
    @pytest.fixture(scope="session")
    def fit_result(self) -> FitResult:
        return FitResult(
            minimum_valid=True,
            execution_time=1.0,
            function_calls=50,
            estimator_value=-2e4,
            parameter_values={
                R"\pi": 3.14,
                "a": -5.0,
                "complex": 1 + 1j,
            },
        )

    def test_count_number_of_parameters(self, fit_result: FitResult):
        assert fit_result.count_number_of_parameters(complex_twice=False) == 3
        assert fit_result.count_number_of_parameters(complex_twice=True) == 4

    def test_pretty_repr(self, fit_result: FitResult):
        src = pretty(fit_result)
        reconstructed = eval(src)
        assert fit_result == reconstructed
