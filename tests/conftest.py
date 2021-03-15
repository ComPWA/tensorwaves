# pylint: disable=redefined-outer-name

from typing import Any, Dict

import expertsystem as es
import pytest
from expertsystem.amplitude.data import EventCollection
from expertsystem.amplitude.dynamics.builder import (
    create_relativistic_breit_wigner_with_ff,
)
from expertsystem.amplitude.helicity import HelicityModel
from expertsystem.amplitude.kinematics import ReactionInfo
from expertsystem.particle import ParticleCollection

from tensorwaves.data import generate_data, generate_phsp
from tensorwaves.data.phasespace import TFUniformRealNumberGenerator
from tensorwaves.data.transform import HelicityTransformer
from tensorwaves.estimator import UnbinnedNLL
from tensorwaves.interfaces import DataSample, DataTransformer
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
def pdg() -> ParticleCollection:
    return es.particle.load_pdg()


@pytest.fixture(scope="session")
def output_dir(pytestconfig) -> str:
    return f"{pytestconfig.rootpath}/tests/output/"


@pytest.fixture(scope="session")
def helicity_model() -> SympyModel:
    model = __create_model(formalism="helicity")
    return SympyModel(
        expression=model.expression,
        parameters=model.parameters,
    )


@pytest.fixture(scope="session")
def canonical_model() -> SympyModel:
    model = __create_model(formalism="canonical-helicity")
    return SympyModel(
        expression=model.expression,
        parameters=model.parameters,
    )


@pytest.fixture(scope="session")
def reaction_info() -> ReactionInfo:
    model = __create_model(formalism="helicity")
    return model.adapter.reaction_info


@pytest.fixture(scope="session")
def kinematics() -> DataTransformer:
    model = __create_model(formalism="helicity")
    return HelicityTransformer(model.adapter)


@pytest.fixture(scope="session")
def phsp_sample(reaction_info: ReactionInfo) -> EventCollection:
    sample = generate_phsp(N_PHSP_EVENTS, reaction_info, random_generator=RNG)
    assert sample.n_events == N_PHSP_EVENTS
    return sample


@pytest.fixture(scope="session")
def phsp_set(
    kinematics: DataTransformer, phsp_sample: EventCollection
) -> DataSample:
    return kinematics.transform(phsp_sample)


@pytest.fixture(scope="session")
def data_sample(
    reaction_info: ReactionInfo,
    kinematics: DataTransformer,
    helicity_model: SympyModel,
) -> EventCollection:
    callable_model = LambdifiedFunction(helicity_model, backend="numpy")
    sample = generate_data(
        N_DATA_EVENTS,
        reaction_info,
        kinematics,
        callable_model,
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
    )


@pytest.fixture(scope="session")
def free_parameters() -> Dict[str, float]:
    return {
        "Gamma_f(0)(500)": 0.3,
        "m_f(0)(980)": 1,
    }


@pytest.fixture(scope="session")
def fit_result(
    estimator: UnbinnedNLL,
    free_parameters: Dict[str, float],
    output_dir: str,
) -> Dict[str, Any]:
    optimizer = Minuit2(
        callback=CallbackList(
            [
                CSVSummary(output_dir + "fit_traceback.csv", step_size=1),
                YAMLSummary(output_dir + "fit_result.yml", step_size=1),
            ]
        )
    )
    return optimizer.optimize(estimator, free_parameters)


def __create_model(formalism: str) -> HelicityModel:
    result = es.generate_transitions(
        initial_state=("J/psi(1S)", [-1, +1]),
        final_state=["gamma", "pi0", "pi0"],
        allowed_intermediate_particles=[
            "f(0)(500)",
            "f(0)(980)",
        ],
        formalism_type=formalism,
        topology_building="isobar",
        allowed_interaction_types=["EM", "strong"],
        number_of_threads=1,
    )
    model_builder = es.amplitude.get_builder(result)
    for name in result.get_intermediate_particles().names:
        model_builder.set_dynamics(
            name, create_relativistic_breit_wigner_with_ff
        )
    return model_builder.generate()
