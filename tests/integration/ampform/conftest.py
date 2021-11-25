# pylint: disable=redefined-outer-name
from pathlib import Path
from typing import Dict

import ampform
import pytest
import qrules
from _pytest.fixtures import SubRequest
from ampform.dynamics.builder import create_relativistic_breit_wigner_with_ff
from ampform.helicity import HelicityModel

from tensorwaves.data import generate_data, generate_phsp
from tensorwaves.data.phasespace import TFUniformRealNumberGenerator
from tensorwaves.data.transform import HelicityTransformer
from tensorwaves.estimator import UnbinnedNLL
from tensorwaves.interface import (
    DataSample,
    DataTransformer,
    FitResult,
    ParameterValue,
)
from tensorwaves.model import LambdifiedFunction, SympyModel
from tensorwaves.optimizer.callbacks import (
    CallbackList,
    CSVSummary,
    YAMLSummary,
)
from tensorwaves.optimizer.minuit import Minuit2


@pytest.fixture(scope="session", params=["canonical", "helicity"])
def reaction(request: SubRequest) -> qrules.ReactionInfo:
    formalism_aliases = {
        "canonical": "canonical-helicity",
        "helicity": "helicity",
    }
    return qrules.generate_transitions(
        initial_state=("J/psi(1S)", [-1, +1]),
        final_state=["gamma", "pi0", "pi0"],
        allowed_intermediate_particles=[
            "f(0)(500)",
            "f(0)(980)",
        ],
        formalism=formalism_aliases[request.param],
        topology_building="isobar",
        allowed_interaction_types=["EM", "strong"],
        number_of_threads=1,
    )


@pytest.fixture(scope="session")
def helicity_model(reaction: qrules.ReactionInfo) -> HelicityModel:
    model_builder = ampform.get_builder(reaction)
    for name in reaction.get_intermediate_particles().names:
        model_builder.set_dynamics(
            name, create_relativistic_breit_wigner_with_ff
        )
    return model_builder.formulate()


@pytest.fixture(scope="session", params=["lambdify", "optimized_lambdify"])
def sympy_model(
    helicity_model: HelicityModel, request: SubRequest
) -> SympyModel:
    max_complexity = None
    if request.param == "optimized_lambdify":
        max_complexity = 200
    return SympyModel(
        expression=helicity_model.expression.doit(),
        parameters=helicity_model.parameter_defaults,
        max_complexity=max_complexity,
    )


@pytest.fixture(scope="session")
def kinematics(helicity_model: HelicityModel) -> DataTransformer:
    return HelicityTransformer(helicity_model.adapter)


@pytest.fixture(scope="session")
def phsp_sample(reaction: qrules.ReactionInfo) -> DataSample:
    n_events = int(1e5)
    initial_state = reaction.initial_state
    final_state = reaction.final_state
    rng = TFUniformRealNumberGenerator(seed=0)
    sample = generate_phsp(
        size=n_events,
        initial_state_mass=initial_state[-1].mass,
        final_state_masses={i: p.mass for i, p in final_state.items()},
        random_generator=rng,
    )
    assert all(map(lambda v: len(v) == n_events, sample.values()))
    return sample


@pytest.fixture(scope="session")
def phsp_set(
    kinematics: DataTransformer, phsp_sample: DataSample
) -> DataSample:
    return kinematics.transform(phsp_sample)


@pytest.fixture(scope="session")
def intensity(sympy_model: SympyModel) -> LambdifiedFunction:
    return LambdifiedFunction(sympy_model, backend="numpy")


@pytest.fixture(scope="session")
def data_sample(
    reaction: qrules.ReactionInfo,
    kinematics: DataTransformer,
    intensity: LambdifiedFunction,
) -> DataSample:
    n_events = int(1e4)
    initial_state = reaction.initial_state
    final_state = reaction.final_state
    rng = TFUniformRealNumberGenerator(seed=0)
    sample = generate_data(
        size=n_events,
        initial_state_mass=initial_state[-1].mass,
        final_state_masses={i: p.mass for i, p in final_state.items()},
        data_transformer=kinematics,
        intensity=intensity,
        random_generator=rng,
    )
    assert all(map(lambda v: len(v) == n_events, sample.values()))
    return sample


@pytest.fixture(scope="session")
def data_set(
    kinematics: DataTransformer,
    data_sample: DataSample,
) -> DataSample:
    return kinematics.transform(data_sample)


@pytest.fixture(scope="session")
def estimator(
    sympy_model: SympyModel, data_set: DataSample, phsp_set: DataSample
) -> UnbinnedNLL:
    return UnbinnedNLL(
        sympy_model,
        dict(data_set),
        dict(phsp_set),
        backend="jax",
    )


@pytest.fixture(scope="session")
def free_parameters(
    reaction: qrules.ReactionInfo,
) -> Dict[str, ParameterValue]:
    # pylint: disable=line-too-long
    if reaction.formalism == "canonical-helicity":
        coefficient_name = (
            R"C_{J/\psi(1S) \xrightarrow[S=1]{L=0} f_{0}(500) \gamma;"
            R" f_{0}(500) \xrightarrow[S=0]{L=0} \pi^{0} \pi^{0}}"
        )
    else:
        coefficient_name = (
            R"C_{J/\psi(1S) \to f_{0}(980)_{0} \gamma_{+1}; f_{0}(980) \to"
            R" \pi^{0}_{0} \pi^{0}_{0}}"
        )
    return {
        coefficient_name: 1.0 + 0.0j,
        "Gamma_f(0)(500)": 0.3,
        "m_f(0)(980)": 1,
    }


@pytest.fixture(scope="session")
def fit_result(
    reaction: qrules.ReactionInfo,
    estimator: UnbinnedNLL,
    free_parameters: Dict[str, float],
    output_dir: Path,
) -> FitResult:
    formalism = reaction.formalism[:4]
    optimizer = Minuit2(
        callback=CallbackList(
            [
                CSVSummary(output_dir / f"fit_traceback_{formalism}.csv"),
                YAMLSummary(
                    output_dir / f"fit_result_{formalism}.yml", step_size=1
                ),
            ]
        )
    )
    return optimizer.optimize(estimator, free_parameters)
