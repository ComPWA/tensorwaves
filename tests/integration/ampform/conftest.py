# pylint: disable=import-outside-toplevel, redefined-outer-name
from typing import TYPE_CHECKING

import pytest
from _pytest.fixtures import SubRequest

from tensorwaves.data import generate_data, generate_phsp
from tensorwaves.data.phasespace import TFUniformRealNumberGenerator
from tensorwaves.data.transform import HelicityTransformer
from tensorwaves.interface import DataSample, DataTransformer
from tensorwaves.model import LambdifiedFunction, SympyModel

if TYPE_CHECKING:
    from ampform.helicity import HelicityModel
    from qrules import ReactionInfo


@pytest.fixture(scope="session", params=["canonical", "helicity"])
def reaction(request: SubRequest) -> "ReactionInfo":
    import qrules

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
def helicity_model(reaction: "ReactionInfo") -> "HelicityModel":
    import ampform
    from ampform.dynamics.builder import (
        create_relativistic_breit_wigner_with_ff,
    )

    model_builder = ampform.get_builder(reaction)
    for name in reaction.get_intermediate_particles().names:
        model_builder.set_dynamics(
            name, create_relativistic_breit_wigner_with_ff
        )
    return model_builder.formulate()


@pytest.fixture(scope="session", params=["lambdify", "optimized_lambdify"])
def sympy_model(
    helicity_model: "HelicityModel", request: SubRequest
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
def kinematics(helicity_model: "HelicityModel") -> DataTransformer:
    return HelicityTransformer(helicity_model.adapter)


@pytest.fixture(scope="session")
def phsp_sample(reaction: "ReactionInfo") -> DataSample:
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
    reaction: "ReactionInfo",
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
