"""`.Function` Adapter for `sympy`-based models."""

from typing import Callable, Dict, FrozenSet, Mapping, Optional, Tuple, Union

import numpy as np
import sympy as sp

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
        if backend == "numpy":
            return np.__dict__

    return backend


class LambdifiedFunction(Function):
    def __init__(
        self,
        function: Callable,
        argument_keys: Tuple[str, ...],
        parameters: Optional[Mapping[str, Union[complex, float]]] = None,
    ) -> None:
        """Wrapper around a callable produced by `~sympy.utilities.lambdify.lambdify`.

        Args:
            function: A callable with **positional arguments** that has been
                created by `~sympy.utilities.lambdify.lambdify`.

            argument_keys: Ordered `tuple` of keys for a
                `~expertsystem.amplitude.data.DataSet` and/or parameter mapping
                the values of which are to be mapped onto the positional
                arguments of the function.

            parameters: Mapping of parameters to their initial values.

        For more info about the intention of this class, see `.Function`.
        """
        if not callable(function):
            raise TypeError("Function argument is not callable")
        self.__callable = function
        function = getattr(function, "__wrapped__", function)  # jit
        if len(argument_keys) != len(function.__code__.co_varnames):
            raise ValueError(
                f"Not all {len(function.__code__.co_varnames)} variables of the"
                f" function {function} are covered by {argument_keys}",
            )
        self.__argument_keys = argument_keys
        self.__parameters: Dict[str, Union[complex, float]] = dict()
        if parameters is not None:
            self.update_parameters(parameters)

    def __call__(self, dataset: DataSample) -> np.ndarray:
        dataset_keys = set(dataset)
        parameter_keys = set(self.parameters)
        argument_keys = set(self.__argument_keys)
        if not argument_keys <= (dataset_keys | parameter_keys):
            missing_keys = argument_keys ^ (
                (argument_keys & dataset_keys)
                | (argument_keys & parameter_keys)
            )
            raise ValueError(
                "Keys of dataset and parameter mapping do not cover all "
                f"function arguments. Missing argument keys: {missing_keys}."
            )
        return self.__callable(
            *[
                dataset[k] if k in dataset else self.__parameters[k]
                for k in self.__argument_keys
            ]
        )

    @property
    def parameters(self) -> Dict[str, Union[float, complex]]:
        return self.__parameters

    def update_parameters(
        self, new_parameters: Mapping[str, Union[float, complex]]
    ) -> None:
        if not set(new_parameters) <= set(self.__argument_keys):
            parameter_keys = set(new_parameters)
            variable_keys = set(self.__argument_keys)
            over_defined = parameter_keys ^ (variable_keys & parameter_keys)
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
    `~expertsystem.amplitude.kinematics.HelicityAdapter` instance is needed to
    convert four momentum information into the custom set of kinematic
    variables.

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
        if not all(map(lambda p: isinstance(p, sp.Symbol), parameters)):
            raise TypeError(f"Not all parameters are of type {sp.Symbol}")

    def lambdify(self, backend: Union[str, tuple, dict]) -> LambdifiedFunction:
        """Lambdify the model using `~sympy.utilities.lambdify.lambdify`."""
        # pylint: disable=import-outside-toplevel
        ordered_symbols = tuple(self.__variables) + tuple(self.__parameters)

        def jax_lambdify() -> Callable:
            from jax import jit

            return jit(
                sp.lambdify(
                    ordered_symbols,
                    self.__expression,
                    modules=backend_modules,
                )
            )

        def numba_lambdify() -> Callable:
            # pylint: disable=import-error
            from numba import jit

            return jit(
                sp.lambdify(
                    ordered_symbols,
                    self.__expression,
                    modules="numpy",
                ),
                parallel=True,
                forceobj=True,
            )

        backend_modules = get_backend_modules(backend)
        full_function: Optional[Callable] = None
        if isinstance(backend, str):
            if backend == "jax":
                full_function = jax_lambdify()
            if backend == "numba":
                full_function = numba_lambdify()
        elif isinstance(backend, tuple):
            if any("jax" in x.__name__ for x in backend):
                full_function = jax_lambdify()
            if any("numba" in x.__name__ for x in backend):
                full_function = numba_lambdify()
        if full_function is None:  # default fallback
            full_function = sp.lambdify(
                ordered_symbols,
                self.__expression,
                modules=backend_modules,
            )
        return LambdifiedFunction(
            full_function,
            argument_keys=tuple(s.name for s in ordered_symbols),
            parameters=self.parameters,
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
