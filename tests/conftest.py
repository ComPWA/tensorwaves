# pylint: disable=redefined-outer-name

from copy import deepcopy

import expertsystem as es
import numpy as np
import pytest
from expertsystem.amplitude.model import AmplitudeModel
from expertsystem.particle import ParticleCollection

from tensorwaves.data.generate import generate_data, generate_phsp
from tensorwaves.data.tf_phasespace import TFUniformRealNumberGenerator
from tensorwaves.estimator import UnbinnedNLL
from tensorwaves.optimizer.callbacks import (
    CallbackList,
    CSVSummary,
    YAMLSummary,
)
from tensorwaves.optimizer.minuit import Minuit2
from tensorwaves.physics.helicity_formalism.amplitude import (
    IntensityBuilder,
    IntensityTF,
)
from tensorwaves.physics.helicity_formalism.kinematics import (
    HelicityKinematics,
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
def helicity_model() -> AmplitudeModel:
    return __create_model(formalism="helicity")


@pytest.fixture(scope="session")
def canonical_model() -> AmplitudeModel:
    return __create_model(formalism="canonical-helicity")


@pytest.fixture(scope="session")
def kinematics(helicity_model: AmplitudeModel) -> HelicityKinematics:
    return HelicityKinematics.from_model(helicity_model)


@pytest.fixture(scope="session")
def phsp_sample(kinematics: HelicityKinematics) -> np.ndarray:
    return generate_phsp(N_PHSP_EVENTS, kinematics, random_generator=RNG)


@pytest.fixture(scope="session")
def phsp_set(kinematics: HelicityKinematics, phsp_sample: np.ndarray) -> dict:
    return kinematics.convert(phsp_sample)


@pytest.fixture(scope="session")
def intensity(
    helicity_model: AmplitudeModel,
    kinematics: HelicityKinematics,
    phsp_sample: np.ndarray,
) -> IntensityTF:
    # https://github.com/ComPWA/tensorwaves/issues/171
    model = deepcopy(helicity_model)
    builder = IntensityBuilder(model.particles, kinematics, phsp_sample)
    return builder.create_intensity(model)


@pytest.fixture(scope="session")
def data_sample(
    kinematics: HelicityKinematics,
    intensity: IntensityTF,
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
    intensity: IntensityTF, data_set: dict, phsp_set: dict
) -> UnbinnedNLL:
    return UnbinnedNLL(intensity, data_set, phsp_set)


@pytest.fixture(scope="session")
def free_parameters() -> dict:
    return {
        "Width_f(0)(500)": 0.3,
        "Position_f(0)(980)": 1,
    }


@pytest.fixture(scope="session")
def fit_result(
    estimator: UnbinnedNLL, free_parameters: dict, output_dir: str
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


def __create_model(formalism: str) -> AmplitudeModel:
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
    model = es.generate_amplitudes(result)
    for name in result.get_intermediate_particles().names:
        model.dynamics.set_breit_wigner(name)
    return model
