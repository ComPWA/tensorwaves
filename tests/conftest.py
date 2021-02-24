# pylint: disable=redefined-outer-name

import expertsystem as es
import numpy as np
import pytest
from expertsystem.amplitude.dynamics.builder import (
    create_relativistic_breit_wigner_with_ff,
)
from expertsystem.amplitude.helicity import ParameterProperties
from expertsystem.particle import ParticleCollection

from tensorwaves.data.generate import generate_data, generate_phsp
from tensorwaves.data.tf_phasespace import TFUniformRealNumberGenerator
from tensorwaves.estimator import SympyUnbinnedNLL
from tensorwaves.optimizer.callbacks import (
    CallbackList,
    CSVSummary,
    YAMLSummary,
)
from tensorwaves.optimizer.minuit import Minuit2
from tensorwaves.physics.amplitude import Intensity, SympyModel
from tensorwaves.physics.helicity_formalism.kinematics import (
    HelicityKinematics,
    ParticleReactionKinematicsInfo,
    SubSystem,
)

N_PHSP_EVENTS = int(1e5)
N_DATA_EVENTS = int(1e4)
RNG = TFUniformRealNumberGenerator(seed=0)


@pytest.fixture(scope="session")
def pdg() -> ParticleCollection:
    return es.io.load_pdg()


@pytest.fixture(scope="session")
def output_dir(pytestconfig) -> str:
    return f"{pytestconfig.rootpath}/tests/output/"


@pytest.fixture(scope="session")
def helicity_model() -> SympyModel:
    return __create_model(formalism="helicity")


@pytest.fixture(scope="session")
def canonical_model() -> SympyModel:
    return __create_model(formalism="canonical-helicity")


@pytest.fixture(scope="session")
def kinematics(pdg: ParticleCollection) -> HelicityKinematics:
    # hardcoding the kinematics here until it has been successfully ported to
    # the expertsystem
    reaction_info = ParticleReactionKinematicsInfo(
        initial_state_names=["J/psi(1S)"],
        final_state_names=["gamma", "pi0", "pi0"],
        particles=pdg,
        fs_id_event_pos_mapping={2: 0, 3: 1, 4: 2},
    )
    kinematics = HelicityKinematics(reaction_info)
    kinematics.register_subsystem(
        SubSystem(
            final_states=[[3, 4], [2]], recoil_state=[], parent_recoil_state=[]
        )
    )
    kinematics.register_subsystem(
        SubSystem(
            final_states=[[3], [4]], recoil_state=[2], parent_recoil_state=[]
        )
    )
    return kinematics


@pytest.fixture(scope="session")
def phsp_sample(kinematics: HelicityKinematics) -> np.ndarray:
    return generate_phsp(N_PHSP_EVENTS, kinematics, random_generator=RNG)


@pytest.fixture(scope="session")
def phsp_set(kinematics: HelicityKinematics, phsp_sample: np.ndarray) -> dict:
    return kinematics.convert(phsp_sample)


@pytest.fixture(scope="session")
def intensity(
    helicity_model: SympyModel,
) -> Intensity:
    return Intensity(helicity_model)


@pytest.fixture(scope="session")
def data_sample(
    kinematics: HelicityKinematics,
    intensity: Intensity,
) -> np.ndarray:
    return generate_data(
        N_DATA_EVENTS, kinematics, intensity, random_generator=RNG
    )


@pytest.fixture(scope="session")
def data_set(
    kinematics: HelicityKinematics,
    data_sample: np.ndarray,
) -> dict:
    return kinematics.convert(data_sample)


@pytest.fixture(scope="session")
def estimator(
    helicity_model: SympyModel, data_set: dict, phsp_set: dict
) -> SympyUnbinnedNLL:
    return SympyUnbinnedNLL(helicity_model, data_set, phsp_set)


@pytest.fixture(scope="session")
def free_parameters() -> dict:
    return {
        "Gamma_f(0)(500)": 0.3,
        "m_f(0)(980)": 1,
    }


@pytest.fixture(scope="session")
def fit_result(
    estimator: SympyUnbinnedNLL, free_parameters: dict, output_dir: str
) -> dict:
    optimizer = Minuit2(
        callback=CallbackList(
            [
                CSVSummary(output_dir + "fit_traceback.csv", step_size=1),
                YAMLSummary(output_dir + "fit_result.yml", step_size=1),
            ]
        )
    )
    return optimizer.optimize(estimator, free_parameters)


def __create_model(formalism: str) -> SympyModel:
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
    model = model_builder.generate()
    return SympyModel(
        expression=model.expression,
        parameters={
            k: v.value if isinstance(v, ParameterProperties) else v
            for k, v in model.parameters.items()
        },
        variables={},
    )
