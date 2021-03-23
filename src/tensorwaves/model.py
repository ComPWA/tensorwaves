"""Evaluateable physics models for amplitude analysis.

The `.model` module takes care of lambdifying mathematical expressions to
computational backends. Currently, mathematical expressions are implemented
as `sympy` expressions only.
"""

from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import sympy as sp
from expertsystem.amplitude.helicity import HelicityModel

from tensorwaves.interfaces import DataSample, Function, Model


def get_backend_modules(
    backend: Union[str, tuple, dict],
) -> Union[str, tuple, dict]:
    """Preprocess the backend argument passed to `~sympy.utilities.lambdify.lambdify`.

    Note in `~sympy.utilities.lambdify.lambdify` the backend is specified via
    the :code:`modules` argument.
    """
    # pylint: disable=import-outside-toplevel
    if isinstance(backend, str):
        if backend == "jax":
            from jax import numpy as jnp
            from jax import scipy as jsp
            from jax.config import config

            config.update("jax_enable_x64", True)

            return (jnp, jsp.special)
        if backend in {"numpy", "numba"}:
            return np.__dict__

    return backend


class LambdifiedFunction(Function):
    def __init__(
        self, model: Model, backend: Union[str, tuple, dict] = "numpy"
    ) -> None:
        """Implements `.Function` based on a `.Model` using `~Model.lambdify`."""
        self.__lambdified_model = model.lambdify(backend=backend)
        self.__parameters = model.parameters
        self.__ordered_args = model.argument_order

    def __call__(self, dataset: DataSample) -> np.ndarray:
        return self.__lambdified_model(
            *[
                dataset[var_name]
                if var_name in dataset
                else self.__parameters[var_name]
                for var_name in self.__ordered_args
            ],
        )

    @property
    def parameters(self) -> Dict[str, Union[float, complex]]:
        return self.__parameters

    def update_parameters(
        self, new_parameters: Mapping[str, Union[float, complex]]
    ) -> None:
        if not set(new_parameters) <= set(self.__parameters):
            over_defined = set(new_parameters) ^ set(self.__parameters)
            raise ValueError(
                f"Parameters {over_defined} do not exist in function arguments"
            )
        self.__parameters.update(new_parameters)


class SympyModel(Model):
    r"""Full definition of an arbitrary model based on `sympy`.

    Note that input for particle physics amplitude models are based on four
    momenta. However, for reasons of convenience, some models may define and
    use a distinct set of kinematic variables (e.g. in the helicity formalism:
    angles :math:`\theta` and :math:`\phi`). In this case, a
    `~.interfaces.DataTransformer` instance (adapter) is needed to transform four
    momentum information into the custom set of kinematic variables.

    Args: expression : A sympy expression that contains the complete
        information of the model based on some inputs. The inputs are defined
        via the `~sympy.core.basic.Basic.free_symbols` attribute of the
        `sympy.Expr <sympy.core.expr.Expr>`. parameters: Defines which inputs
        of the model are parameters. The keys represent the parameter set,
        while the values represent their default values. Consequently the
        variables of the model are defined as the intersection of the total
        input set with the parameter set.
    """

    def __init__(
        self,
        expression: sp.Expr,
        parameters: Dict[sp.Symbol, Union[float, complex]],
    ) -> None:
        self.__expression: sp.Expr = expression.doit()
        self.__parameters = parameters
        self.__variables: FrozenSet[sp.Symbol] = frozenset(
            s
            for s in self.__expression.free_symbols
            if s.name not in self.parameters
        )
        self.__argument_order = tuple(self.__variables) + tuple(
            self.__parameters
        )
        if not all(map(lambda p: isinstance(p, sp.Symbol), parameters)):
            raise TypeError(f"Not all parameters are of type {sp.Symbol}")

    def lambdify(self, backend: Union[str, tuple, dict]) -> Callable:
        """Lambdify the model using `~sympy.utilities.lambdify.lambdify`."""
        # pylint: disable=import-outside-toplevel
        ordered_symbols = self.__argument_order

        def jax_lambdify() -> Callable:
            import jax

            return jax.jit(
                sp.lambdify(
                    ordered_symbols,
                    self.__expression,
                    modules=backend_modules,
                )
            )

        def numba_lambdify() -> Callable:
            # pylint: disable=import-error
            import numba

            return numba.jit(
                sp.lambdify(
                    ordered_symbols,
                    self.__expression,
                    modules="numpy",
                ),
                forceobj=True,
                parallel=True,
            )

        backend_modules = get_backend_modules(backend)
        if isinstance(backend, str):
            if backend == "jax":
                return jax_lambdify()
            if backend == "numba":
                return numba_lambdify()
        if isinstance(backend, tuple):
            if any("jax" in x.__name__ for x in backend):
                return jax_lambdify()
            if any("numba" in x.__name__ for x in backend):
                return numba_lambdify()
        return sp.lambdify(
            ordered_symbols,
            self.__expression,
            modules=backend_modules,
        )

    def performance_optimize(self, fix_inputs: DataSample) -> "Model":
        raise NotImplementedError

    @property
    def parameters(self) -> Dict[str, Union[float, complex]]:
        return {
            symbol.name: value for symbol, value in self.__parameters.items()
        }

    @property
    def variables(self) -> FrozenSet[str]:
        """Expected input variable names."""
        return frozenset({symbol.name for symbol in self.__variables})

    @property
    def argument_order(self) -> Tuple[str, ...]:
        return tuple(x.name for x in self.__argument_order)


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
        parameters=model.parameters,
    )
    return LambdifiedFunction(sympy_model, backend=backend)
