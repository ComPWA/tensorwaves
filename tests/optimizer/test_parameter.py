import pytest

from tensorwaves.optimizer.parameter import ParameterFlattener


def describe_ParameterFlattener():
    @pytest.fixture(scope="module")
    def parameter_flattener() -> ParameterFlattener:
        return ParameterFlattener({"var1": 1 + 0j, "var2": 2})

    @pytest.mark.parametrize(
        ("unflattened_parameters", "expected_flattened_parameters"),
        [
            (
                {"var1": 0.5 + 2j, "var2": -1.2},
                {"real_var1": 0.5, "imag_var1": 2.0, "var2": -1.2},
            ),
            (
                {"var1": 0.5 - 6.4j, "var2": -1.2},
                {"real_var1": 0.5, "imag_var1": -6.4, "var2": -1.2},
            ),
        ],
    )
    def it_splits_a_complex_parameter_into_a_real_and_imaginary_part(
        parameter_flattener, unflattened_parameters, expected_flattened_parameters
    ):
        assert (
            parameter_flattener.flatten(unflattened_parameters)
            == expected_flattened_parameters
        )

    @pytest.mark.parametrize(
        ("flattened_parameters", "expected_unflattened_parameters"),
        [
            (
                {"real_var1": 0.5, "imag_var1": 2.0, "var2": -1.2},
                {"var1": 0.5 + 2j, "var2": -1.2},
            ),
            (
                {"real_var1": 0.5, "imag_var1": -6.4, "var2": -1.2},
                {"var1": 0.5 - 6.4j, "var2": -1.2},
            ),
        ],
    )
    def it_recombines_a_real_and_imaginary_part_into_a_complex_parameter(
        parameter_flattener, flattened_parameters, expected_unflattened_parameters
    ):
        assert (
            parameter_flattener.unflatten(flattened_parameters)
            == expected_unflattened_parameters
        )
