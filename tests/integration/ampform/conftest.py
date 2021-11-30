# pylint: disable=redefined-outer-name
from typing import Tuple

import ampform
import pytest
import qrules
from _pytest.fixtures import SubRequest
from ampform.dynamics.builder import create_relativistic_breit_wigner_with_ff
from ampform.helicity import HelicityModel

from tensorwaves.data import generate_data, generate_phsp
from tensorwaves.data.phasespace import TFUniformRealNumberGenerator
from tensorwaves.data.transform import HelicityTransformer
from tensorwaves.function import LambdifiedFunction
from tensorwaves.function.sympy import create_function
from tensorwaves.interface import DataSample, DataTransformer


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
    return kinematics(phsp_sample)


@pytest.fixture(
    scope="session",
    params=[
        ("jax", False),
        ("jax", True),
    ],
    ids=[
        "jax-normal",
        "jax-fast",
    ],
)
def function_fixture(
    helicity_model: HelicityModel, request: SubRequest
) -> Tuple[LambdifiedFunction, str]:
    backend, fast_lambdify = request.param
    if fast_lambdify:
        max_complexity = None
        lambdify_type = f"{backend}-fast"
    else:
        max_complexity = 200
        lambdify_type = f"{backend}-normal"
    function = create_function(
        expression=helicity_model.expression.doit(),
        parameters=helicity_model.parameter_defaults,
        max_complexity=max_complexity,
        backend=backend,
    )
    return function, lambdify_type


@pytest.fixture(scope="session")
def data_sample(
    reaction: qrules.ReactionInfo,
    kinematics: DataTransformer,
    function_fixture: Tuple[LambdifiedFunction, str],
) -> DataSample:
    function, _ = function_fixture
    n_events = int(1e4)
    initial_state = reaction.initial_state
    final_state = reaction.final_state
    rng = TFUniformRealNumberGenerator(seed=0)
    sample = generate_data(
        size=n_events,
        initial_state_mass=initial_state[-1].mass,
        final_state_masses={i: p.mass for i, p in final_state.items()},
        data_transformer=kinematics,
        intensity=function,
        random_generator=rng,
    )
    assert all(map(lambda v: len(v) == n_events, sample.values()))
    return sample


@pytest.fixture(scope="session")
def data_set(
    kinematics: DataTransformer,
    data_sample: DataSample,
) -> DataSample:
    return kinematics(data_sample)
