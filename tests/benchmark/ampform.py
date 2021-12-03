# pylint: disable=no-self-use
from typing import List, Mapping, Optional, Sequence, Tuple

import ampform
import pytest
import qrules
from ampform.dynamics.builder import create_relativistic_breit_wigner_with_ff
from ampform.helicity import HelicityModel, ParameterValue
from qrules.combinatorics import StateDefinition

import tensorwaves as tw
from tensorwaves.data.phasespace import TFUniformRealNumberGenerator
from tensorwaves.data.transform import HelicityTransformer
from tensorwaves.function.sympy import create_parametrized_function
from tensorwaves.interface import DataSample, FitResult, ParametrizedFunction


def formulate_amplitude_model(
    formalism: str,
    initial_state: StateDefinition,
    final_state: Sequence[StateDefinition],
    intermediate_states: Optional[List[str]] = None,
    interaction_types: Optional[List[str]] = None,
) -> HelicityModel:
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
    model: HelicityModel, backend: str, max_complexity: Optional[int] = None
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
) -> Tuple[DataSample, DataSample]:
    reaction = model.adapter.reaction_info
    final_state = reaction.final_state
    rng = TFUniformRealNumberGenerator(seed=0)
    phsp_sample = tw.data.generate_phsp(
        size=phsp_sample_size,
        initial_state_mass=reaction.initial_state[-1].mass,
        final_state_masses={i: p.mass for i, p in final_state.items()},
        random_generator=rng,
    )

    helicity_transformer = HelicityTransformer(model.adapter)
    data_sample = tw.data.generate_data(
        size=data_sample_size,
        initial_state_mass=reaction.initial_state[-1].mass,
        final_state_masses={i: p.mass for i, p in final_state.items()},
        data_transformer=helicity_transformer,
        intensity=function,
        random_generator=rng,
    )

    data_set = helicity_transformer(data_sample)
    phsp_set = helicity_transformer(phsp_sample)
    return data_set, phsp_set


def fit(
    data_set: DataSample,
    phsp_set: DataSample,
    function: ParametrizedFunction,
    initial_parameters: Mapping[str, ParameterValue],
    backend: str,
) -> FitResult:
    estimator = tw.estimator.UnbinnedNLL(
        function,
        data=data_set,
        phsp=phsp_set,
        backend=backend,
    )
    optimizer = tw.optimizer.Minuit2()

    return optimizer.optimize(estimator, initial_parameters)


class TestJPsiToGammaPiPi:
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
    @pytest.mark.parametrize("backend", ["jax"])
    @pytest.mark.parametrize("size", [100, 500, 1000, 5000, 10_000])
    def test_data(self, backend, benchmark, model, size):
        n_data = size
        n_phsp = 10 * n_data
        function = create_function(model, backend)
        data, phsp = benchmark(generate_data, model, function, n_data, n_phsp)
        assert len(next(iter(data.values()))) == n_data
        assert len(next(iter(phsp.values()))) == n_phsp

    @pytest.mark.benchmark(group="fit", min_rounds=1)
    @pytest.mark.parametrize("backend", ["jax"])
    @pytest.mark.parametrize("size", [10_000])
    def test_fit(self, backend, benchmark, model, size):
        n_data = size
        n_phsp = 10 * n_data
        function = create_function(model, backend)
        data, phsp = generate_data(model, function, n_data, n_phsp)

        coefficients = [p for p in function.parameters if p.startswith("C_{")]
        assert len(coefficients) >= 1
        coefficient_name = coefficients[1]
        initial_parameters = {
            coefficient_name: 1.0 + 0.0j,
            "Gamma_f(0)(500)": 0.3,
            "m_f(0)(980)": 1,
        }
        fit_result = benchmark(
            fit, data, phsp, function, initial_parameters, backend
        )
        assert fit_result.minimum_valid
