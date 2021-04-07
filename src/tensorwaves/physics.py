"""Extract physics info from data and fit output."""

from typing import Iterable, Sequence, Union

import sympy as sp
from ampform.helicity import HelicityModel

from tensorwaves.model import LambdifiedFunction, SympyModel


def add_components(
    model: HelicityModel,
    components: Union[str, Iterable[str]],
) -> sp.Expr:
    """Coherently or incoherently add components of a helicity model."""
    if isinstance(components, str):
        components = [components]
    for component in components:
        if component not in model.components:
            raise KeyError(
                f'Component "{component}" not in model components',
                list(model.components),
            )
    if any(map(lambda c: c.startswith("I"), components)) and any(
        map(lambda c: c.startswith("A"), components)
    ):
        intensity_sum = add_components(
            model,
            components=filter(lambda c: c.startswith("I"), components),
        )
        amplitude_sum = add_components(
            model,
            components=filter(lambda c: c.startswith("A"), components),
        )
        return intensity_sum + amplitude_sum
    if all(map(lambda c: c.startswith("I"), components)):
        return sum(model.components[c] for c in components)
    if all(map(lambda c: c.startswith("A"), components)):
        return abs(sum(model.components[c] for c in components)) ** 2
    raise ValueError('Not all component names started with either "A" or "I"')


def create_intensity_component(
    model: HelicityModel,
    components: Union[str, Sequence[str]],
    backend: str,
) -> LambdifiedFunction:
    """Create a `.LambdifiedFunction` of a sum of helicity model components."""
    added_components = add_components(model, components)
    sympy_model = SympyModel(
        expression=added_components,
        parameters=model.parameter_defaults,
    )
    return LambdifiedFunction(sympy_model, backend=backend)
