# pylint: disable=import-outside-toplevel, no-self-use
from pprint import pprint
from typing import TYPE_CHECKING, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pytest

import tensorwaves as tw
from tensorwaves.data import IntensityDistributionGenerator
from tensorwaves.data.phasespace import (
    TFPhaseSpaceGenerator,
    TFUniformRealNumberGenerator,
)
from tensorwaves.data.transform import SympyDataTransformer
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
    initial_state: "StateDefinition",
    final_state: Sequence["StateDefinition"],
    intermediate_states: Optional[List[str]] = None,
    interaction_types: Optional[List[str]] = None,
) -> "HelicityModel":
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
    model: "HelicityModel", backend: str, max_complexity: Optional[int] = None
) -> ParametrizedFunction:
    return create_parametrized_function(
        expression=model.expression.doit(),
        parameters=model.parameter_defaults,
        max_complexity=max_complexity,
        backend=backend,
    )


def generate_data(
    model: "HelicityModel",
    function: ParametrizedFunction,
    data_sample_size: int,
    phsp_sample_size: int,
    backend: str,
    transform: bool = False,
) -> Tuple[DataSample, DataSample]:
    reaction = model.reaction_info
    final_state = reaction.final_state
    expressions = model.kinematic_variables
    converter = SympyDataTransformer.from_sympy(expressions, backend)

    rng = TFUniformRealNumberGenerator(seed=0)
    phsp_generator = TFPhaseSpaceGenerator(
        initial_state_mass=reaction.initial_state[-1].mass,
        final_state_masses={i: p.mass for i, p in final_state.items()},
    )
    data_generator = IntensityDistributionGenerator(
        phsp_generator, function, transformer=converter
    )
    phsp = phsp_generator.generate(phsp_sample_size, rng)
    data = data_generator.generate(data_sample_size, rng)

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
            [1.30998577367, -1.11343003896, -0.43080737645, -0.53920430261],
            [1.39285735587, -0.8023158638, 0.17870059434, -1.12445852128],
            [1.48105543582, 0.81816399518, -1.04574090056, 0.65616983309],
            [1.50811690859, -0.71505787232, 1.31656646219, -0.17251550611],
            [1.50110691532, 0.09839660888, 1.28437745681, -0.77072344391],
        ],
        "p1": [
            [0.43494801621, -0.08231783732, 0.39797088163, -0.07618393074],
            [0.80478644078, -0.01366453826, -0.06367749624, 0.79070913463],
            [1.20741065761, -0.47126047814, 0.98365647532, -0.49995525787],
            [0.79092678569, 0.32563849669, -0.64424028146, 0.29370133299],
            [1.24457660001, -0.17602687045, -1.11572774371, 0.50489436476],
        ],
        "p2": [
            [1.35196621012, 1.19574787628, 0.03283649482, 0.61538823335],
            [0.89925620335, 0.81598040206, -0.11502309811, 0.33374938665],
            [0.40843390657, -0.34690351704, 0.06208442523, -0.15621457522],
            [0.79785630572, 0.38941937563, -0.67232618072, -0.12118582688],
            [0.35121648467, 0.07763026157, -0.16864971311, 0.26582907915],
        ],
    }

    @pytest.fixture(scope="session")
    def model(self) -> "HelicityModel":
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
            "Gamma_f(0)(500)": 0.3,
            "m_f(0)(980)": 1,
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
