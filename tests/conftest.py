# pylint: disable=redefined-outer-name
from typing import Dict

import ampform
import pytest
import qrules
from ampform.data import EventCollection
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

N_PHSP_EVENTS = int(1e5)
N_DATA_EVENTS = int(1e4)
RNG = TFUniformRealNumberGenerator(seed=0)


@pytest.fixture(scope="session")
def pdg() -> qrules.ParticleCollection:
    return qrules.particle.load_pdg()


@pytest.fixture(scope="session")
def output_dir(pytestconfig) -> str:
    return f"{pytestconfig.rootpath}/tests/output/"


@pytest.fixture(scope="session")
def canonical_reaction() -> qrules.ReactionInfo:
    return __generate_reaction(formalism="canonical-helicity")


@pytest.fixture(scope="session")
def reaction() -> qrules.ReactionInfo:
    return __generate_reaction(formalism="helicity")


@pytest.fixture(scope="session")
def canonical_model(canonical_reaction: qrules.ReactionInfo) -> SympyModel:
    model = __formulate_model(canonical_reaction)
    return SympyModel(
        expression=model.expression,
        parameters=model.parameter_defaults,
    )


@pytest.fixture(scope="session")
def es_helicity_model(reaction: qrules.ReactionInfo) -> HelicityModel:
    return __formulate_model(reaction)


@pytest.fixture(scope="session")
def helicity_model(es_helicity_model: HelicityModel) -> SympyModel:
    model = es_helicity_model
    return SympyModel(
        expression=model.expression,
        parameters=model.parameter_defaults,
    )


@pytest.fixture(scope="session")
def kinematics(es_helicity_model: HelicityModel) -> DataTransformer:
    model = es_helicity_model
    return HelicityTransformer(model.adapter)


@pytest.fixture(scope="session")
def phsp_sample(reaction: qrules.ReactionInfo) -> EventCollection:
    initial_state = reaction.initial_state
    final_state = reaction.final_state
    sample = generate_phsp(
        N_PHSP_EVENTS,
        initial_state_mass=initial_state[-1].mass,
        final_state_masses={i: p.mass for i, p in final_state.items()},
        random_generator=RNG,
    )
    assert sample.n_events == N_PHSP_EVENTS
    return sample


@pytest.fixture(scope="session")
def phsp_set(
    kinematics: DataTransformer, phsp_sample: EventCollection
) -> DataSample:
    return kinematics.transform(phsp_sample)


@pytest.fixture(scope="session")
def intensity(helicity_model: SympyModel) -> LambdifiedFunction:
    return LambdifiedFunction(helicity_model, backend="numpy")


@pytest.fixture(scope="session")
def data_sample(
    reaction: qrules.ReactionInfo,
    kinematics: DataTransformer,
    intensity: LambdifiedFunction,
) -> EventCollection:
    initial_state = reaction.initial_state
    final_state = reaction.final_state
    sample = generate_data(
        N_DATA_EVENTS,
        initial_state_mass=initial_state[-1].mass,
        final_state_masses={i: p.mass for i, p in final_state.items()},
        data_transformer=kinematics,
        intensity=intensity,
        random_generator=RNG,
    )
    assert sample.n_events == N_DATA_EVENTS
    return sample


@pytest.fixture(scope="session")
def data_set(
    kinematics: DataTransformer,
    data_sample: EventCollection,
) -> DataSample:
    return kinematics.transform(data_sample)


@pytest.fixture(scope="session")
def estimator(
    helicity_model: SympyModel, data_set: DataSample, phsp_set: DataSample
) -> UnbinnedNLL:
    return UnbinnedNLL(
        helicity_model,
        dict(data_set),
        dict(phsp_set),
        backend="jax",
    )


@pytest.fixture(scope="session")
def free_parameters() -> Dict[str, ParameterValue]:
    # pylint: disable=line-too-long
    return {
        R"C_{J/\psi(1S) \to f_{0}(980)_{0} \gamma_{+1}; f_{0}(980) \to \pi^{0}_{0} \pi^{0}_{0}}": 1.0
        + 0.0j,
        "Gamma_f(0)(500)": 0.3,
        "m_f(0)(980)": 1,
    }


@pytest.fixture(scope="session")
def fit_result(
    estimator: UnbinnedNLL,
    free_parameters: Dict[str, float],
    output_dir: str,
) -> FitResult:
    optimizer = Minuit2(
        callback=CallbackList(
            [
                CSVSummary(output_dir + "fit_traceback.csv"),
                YAMLSummary(output_dir + "fit_result.yml", step_size=1),
            ]
        )
    )
    return optimizer.optimize(estimator, free_parameters)


def __generate_reaction(formalism: str) -> qrules.ReactionInfo:
    return qrules.generate_transitions(
        initial_state=("J/psi(1S)", [-1, +1]),
        final_state=["gamma", "pi0", "pi0"],
        allowed_intermediate_particles=[
            "f(0)(500)",
            "f(0)(980)",
        ],
        formalism=formalism,
        topology_building="isobar",
        allowed_interaction_types=["EM", "strong"],
        number_of_threads=1,
    )


def __formulate_model(reaction: qrules.ReactionInfo) -> HelicityModel:
    model_builder = ampform.get_builder(reaction)
    for name in reaction.get_intermediate_particles().names:
        model_builder.set_dynamics(
            name, create_relativistic_breit_wigner_with_ff
        )
    return model_builder.formulate()
