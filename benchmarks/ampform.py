# pylint: disable=import-outside-toplevel
from __future__ import annotations

from pprint import pprint
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import pytest

import tensorwaves as tw
from tensorwaves.data import (
    IntensityDistributionGenerator,
    SympyDataTransformer,
    TFPhaseSpaceGenerator,
    TFUniformRealNumberGenerator,
    TFWeightedPhaseSpaceGenerator,
)
from tensorwaves.function.sympy import create_parametrized_function
from tensorwaves.interface import (
    DataSample,
    FitResult,
    ParameterValue,
    ParametrizedFunction,
)

if TYPE_CHECKING:
    from ampform.helicity import HelicityModel
    from qrules.combinatorics import StateDefinition


def formulate_amplitude_model(
    formalism: str,
    initial_state: StateDefinition,
    final_state: Sequence[StateDefinition],
    intermediate_states: list[str] | None = None,
    interaction_types: list[str] | None = None,
) -> HelicityModel:
    import ampform
    import qrules
    from ampform.dynamics.builder import (
        create_relativistic_breit_wigner_with_ff,
    )

    reaction = qrules.generate_transitions(
        initial_state=initial_state,
        final_state=final_state,
        allowed_intermediate_particles=intermediate_states,
        formalism=formalism,
        allowed_interaction_types=interaction_types,
    )

    builder = ampform.get_builder(reaction)
    for name in reaction.get_intermediate_particles().names:
        builder.set_dynamics(name, create_relativistic_breit_wigner_with_ff)
    return builder.formulate()


def create_function(
    model: HelicityModel, backend: str, max_complexity: int | None = None
) -> ParametrizedFunction:
    return create_parametrized_function(
        expression=model.expression.doit(),
        parameters=model.parameter_defaults,
        max_complexity=max_complexity,
        backend=backend,
    )


def generate_data(
    model: HelicityModel,
    function: ParametrizedFunction,
    data_sample_size: int,
    phsp_sample_size: int,
    backend: str,
    transform: bool = False,
) -> tuple[DataSample, DataSample]:
    # pylint: disable=too-many-locals
    reaction = model.reaction_info
    final_state = reaction.final_state
    expressions = model.kinematic_variables
    converter = SympyDataTransformer.from_sympy(expressions, backend)

    initial_state_mass = reaction.initial_state[-1].mass
    final_state_masses = {i: p.mass for i, p in final_state.items()}
    phsp_generator = TFPhaseSpaceGenerator(
        initial_state_mass, final_state_masses
    )
    phsp = phsp_generator.generate(
        phsp_sample_size, rng=TFUniformRealNumberGenerator(seed=0)
    )

    weighted_phsp_generator = TFWeightedPhaseSpaceGenerator(
        initial_state_mass, final_state_masses
    )
    data_generator = IntensityDistributionGenerator(
        weighted_phsp_generator, function, domain_transformer=converter
    )
    data = data_generator.generate(
        data_sample_size, rng=TFUniformRealNumberGenerator(seed=0)
    )

    if transform:
        data = converter(data)
        phsp = converter(phsp)
    return data, phsp


def fit(
    data: DataSample,
    phsp: DataSample,
    function: ParametrizedFunction,
    initial_parameters: Mapping[str, ParameterValue],
    backend: str,
) -> FitResult:
    estimator = tw.estimator.UnbinnedNLL(
        function,
        data=data,
        phsp=phsp,
        backend=backend,
    )
    optimizer = tw.optimizer.Minuit2()
    return optimizer.optimize(estimator, initial_parameters)


class TestJPsiToGammaPiPi:
    expected_data = {
        "p0": [
            [1.50757377596, 0.37918944935, 0.73396599969, 1.26106620078],
            [1.41389525301, -0.07315064441, -0.21998573758, 1.39475985207],
            [1.52128570461, 0.06569896528, -1.51812710851, 0.0726906006],
            [1.51480310845, 1.40672331053, 0.49678572189, -0.26260603856],
            [1.52384281483, 0.79694939592, 1.29832389761, -0.03638188481],
        ],
        "p1": [
            [1.42066087326, -0.34871369761, -0.72119471428, -1.1654765212],
            [0.96610319301, -0.26739932067, -0.15455480956, -0.90539883872],
            [0.60647770024, 0.11616448713, 0.57584161239, -0.06714695611],
            [1.01045883083, -0.88651015826, -0.46024226278, 0.0713099651],
            [1.04324742713, -0.48051670276, -0.91259832182, -0.08009031815],
        ],
        "p2": [
            [0.16866535079, -0.03047575173, -0.01277128542, -0.09558967958],
            [0.71690155399, 0.34054996508, 0.37454054715, -0.48936101336],
            [0.96913659515, -0.18186345241, 0.94228549612, -0.00554364449],
            [0.57163806072, -0.52021315227, -0.03654345912, 0.19129607347],
            [0.52980975805, -0.31643269316, -0.38572557579, 0.11647220296],
        ],
    }

    @pytest.fixture(scope="session")
    def model(self) -> HelicityModel:
        return formulate_amplitude_model(
            formalism="canonical-helicity",
            initial_state=("J/psi(1S)", [-1, +1]),
            final_state=["gamma", "pi0", "pi0"],
            intermediate_states=["f(0)(500)", "f(0)(980)"],
            interaction_types=["EM", "strong"],
        )

    @pytest.mark.benchmark(group="data", min_rounds=1)
    @pytest.mark.parametrize("backend", ["jax", "numpy", "tf"])
    @pytest.mark.parametrize("size", [10_000])
    def test_data(self, backend, benchmark, model, size):
        n_data = size
        n_phsp = 10 * n_data
        function = create_function(model, backend)
        data, phsp = benchmark(
            generate_data, model, function, n_data, n_phsp, backend
        )
        assert len(next(iter(data.values()))) == n_data
        assert len(next(iter(phsp.values()))) == n_phsp

        # test data sample values
        # https://github.com/ComPWA/tensorwaves/blob/b5abfad/tests/integration/ampform/test_data.py
        sample_size = len(next(iter(self.expected_data.values())))
        print_data_sample(data, sample_size)
        for i, expected in self.expected_data.items():
            assert pytest.approx(data[i][:sample_size]) == expected

    @pytest.mark.benchmark(group="fit", min_rounds=1)
    @pytest.mark.parametrize("backend", ["jax"])
    @pytest.mark.parametrize("size", [10_000])
    def test_fit(self, backend, benchmark, model, size):
        n_data = size
        n_phsp = 10 * n_data
        function = create_function(model, backend)
        data, phsp = generate_data(
            model, function, n_data, n_phsp, backend, transform=True
        )

        coefficients = [p for p in function.parameters if p.startswith("C_{")]
        assert len(coefficients) >= 1
        coefficient_name = coefficients[1]
        initial_parameters = {
            coefficient_name: 1.0 + 0.0j,
            "m_{f_{0}(980)}": 1,
            R"\Gamma_{f_{0}(500)}": 0.3,
        }
        fit_result = benchmark(
            fit, data, phsp, function, initial_parameters, backend
        )
        assert fit_result.minimum_valid


def print_data_sample(data: DataSample, sample_size: int) -> None:
    """Print a `.DataSample`, so it can be pasted into the expected sample."""
    print()
    pprint(
        {
            i: np.round(four_momenta[:sample_size], decimals=11).tolist()
            for i, four_momenta in data.items()
        }
    )
