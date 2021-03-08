"""`~.Function` Adapter for `sympy` based models."""

from typing import Any, Callable, Dict, FrozenSet, Optional, Tuple, Union

import sympy as sp

from tensorwaves.interfaces import Function, Model


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
            import numpy as np

            return np.__dict__

    return backend


class SympyModel(Model):
    r"""Full definition of an arbitrary model based on `sympy`.

    Note that input for particle physics amplitude models are based on four
    momenta. However, for reasons of convenience, some models may define and
    use a distinct set of kinematic variables (e.g. in the helicity formalism:
    angles :math:`\theta` and :math:`\phi`). In this case, a `.Kinematics`
    instance (adapter) is needed to convert four momentum information into the
    custom set of kinematic variables.

    Args:
        expression : A sympy expression that contains the complete information
          of the model based on some inputs. The inputs are defined via the
          `~sympy.core.basic.Basic.free_symbols` attribute of the `sympy.Expr
          <sympy.core.expr.Expr>`.
        parameters: Defines which inputs of the model are parameters. The keys
          represent the parameter set, while the values represent their default
          values. Consequently the variables of the model are defined as the
          intersection of the total input set with the parameter set.
    """

    def __init__(
        self,
        expression: sp.Expr,
        parameters: Dict[sp.Symbol, Union[float, complex]],
    ) -> None:
        self.__expression: sp.Expr = expression.doit()
        self.__parameters = parameters
        self.__variables: FrozenSet[sp.Symbol] = frozenset(
            {
                symbol
                for symbol in self.__expression.free_symbols
                if symbol.name not in self.parameters
            }
        )

    def lambdify(self, backend: Union[str, tuple, dict]) -> Function:
        """Lambdify the model using `~sympy.utilities.lambdify.lambdify`."""
        # pylint: disable=import-outside-toplevel
        variables = tuple(self.__expression.free_symbols)

        def jax_lambdify() -> Callable:
            from jax import jit

            return jit(
                sp.lambdify(
                    variables,
                    self.__expression,
                    modules=backend_modules,
                )
            )

        callable_model: Optional[Callable] = None
        if isinstance(backend, str):
            if backend == "jax":
                callable_model = jax_lambdify()
            if backend == "numba":
                # pylint: disable=import-error
                from numba import jit

                callable_model = jit(
                    sp.lambdify(
                        variables,
                        self.__expression,
                        modules="numpy",
                    ),
                    parallel=True,
                )
        elif isinstance(backend, tuple):
            if any("jax" in x.__name__ for x in backend):
                callable_model = jax_lambdify()
        if callable_model is None:  # default
            backend_modules = get_backend_modules(backend)
            callable_model = sp.lambdify(
                variables,
                self.__expression,
                modules=backend_modules,
            )
        if callable_model is None:
            raise ValueError(f"Failed to lambdify model for backend {backend}")

        input_variable_order: Tuple[str, ...] = tuple(
            x.name for x in self.__expression.free_symbols
        )

        def function_wrapper(dataset: Dict[str, Any]) -> Any:
            return callable_model(  # type: ignore
                *(
                    dataset[var_name]
                    if var_name in dataset
                    else self.parameters[var_name]
                    for var_name in input_variable_order
                )
            )

        return function_wrapper

    def performance_optimize(self, fix_inputs: Dict[str, Any]) -> "Model":
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
