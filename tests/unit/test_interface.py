# pylint: disable=no-self-use
from tensorwaves.interface import FitResult


class TestFitResult:
    def test_count_number_of_parameters(self):
        fit_result = FitResult(
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
        assert fit_result.count_number_of_parameters(complex_twice=False) == 3
        assert fit_result.count_number_of_parameters(complex_twice=True) == 4
