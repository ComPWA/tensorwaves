import pytest
from expertsystem import io
from expertsystem.amplitude.canonical_decay import CanonicalAmplitudeGenerator
from expertsystem.amplitude.helicity_decay import HelicityAmplitudeGenerator
from expertsystem.amplitude.model import AmplitudeModel
from expertsystem.particle import ParticleCollection
from expertsystem.reaction import InteractionTypes, StateTransitionManager


@pytest.fixture(scope="session")
def pdg() -> ParticleCollection:
    return io.load_pdg()


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
    stm = StateTransitionManager(
        initial_state=[("J/psi(1S)")],
        final_state=[("gamma"), ("pi0"), ("pi0")],
        allowed_intermediate_particles=["f(0)(980)"],
        formalism_type=formalism,
        topology_building="isobar",
        number_of_threads=1,
    )

    stm.set_allowed_interaction_types([InteractionTypes.Strong])
    graph_interaction_settings_groups = stm.prepare_graphs()
    result = stm.find_solutions(graph_interaction_settings_groups)

    if formalism == "helicity":
        amplitude_generator = HelicityAmplitudeGenerator()
    elif formalism == "canonical-helicity":
        amplitude_generator = CanonicalAmplitudeGenerator()
    else:
        raise NotImplementedError(f"No amplitude generator for {formalism}")

    amplitude_model = amplitude_generator.generate(result)
    return amplitude_model
