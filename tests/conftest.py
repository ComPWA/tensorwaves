import expertsystem as es
import pytest
from expertsystem.amplitude.model import AmplitudeModel
from expertsystem.particle import ParticleCollection


@pytest.fixture(scope="session")
def pdg() -> ParticleCollection:
    return es.io.load_pdg()


@pytest.fixture(scope="session")
def output_dir(pytestconfig) -> str:
    return f"{pytestconfig.rootpath}/tests/output/"


@pytest.fixture(scope="session")
def helicity_model() -> AmplitudeModel:
    return __create_model(formalism="canonical-helicity")


@pytest.fixture(scope="session")
def canonical_model() -> AmplitudeModel:
    return __create_model(formalism="helicity")


def __create_model(formalism: str) -> AmplitudeModel:
    result = es.generate_transitions(
        initial_state=("J/psi(1S)", [-1, +1]),
        final_state=["gamma", "pi0", "pi0"],
        allowed_intermediate_particles=["f(0)(980)"],
        formalism_type=formalism,
        topology_building="isobar",
        allowed_interaction_types=["EM", "strong"],
        number_of_threads=1,
    )
    model = es.generate_amplitudes(result)
    for name in result.get_intermediate_particles().names:
        model.dynamics.set_breit_wigner(name)
    return model
