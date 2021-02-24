"""`~.Function` Adapter for `sympy` based models."""

import logging
from typing import Any, Callable, Dict, Tuple, Union

import attr
import sympy

from tensorwaves.interfaces import Function


@attr.s(frozen=True)
class SympyModel:
    expression: sympy.Expr = attr.ib()
    parameters: Dict[
        sympy.Symbol, Union[float, complex, sympy.Expr]
    ] = attr.ib()
    variables: Dict[sympy.Symbol, sympy.Expr] = attr.ib()


def get_backend_modules(
    backend: Union[str, tuple, dict],
) -> Union[str, tuple, dict]:
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


def lambdify(
    variables: Tuple[sympy.Symbol, ...],
    expression: sympy.Expr,
    backend: Union[str, tuple, dict],
) -> Callable:
    # pylint: disable=import-outside-toplevel
    backend_modules = get_backend_modules(backend)

    def jax_lambdify() -> Callable:
        from jax import jit

        return jit(
            sympy.lambdify(
                variables,
                expression,
                modules=backend_modules,
            )
        )

    if isinstance(backend, str):
        if backend == "jax":
            return jax_lambdify()
        if backend == "numba":
            from numba import jit

            return jit(
                sympy.lambdify(
                    variables,
                    expression,
                    modules="numpy",
                ),
                parallel=True,
            )
    if isinstance(backend, tuple):
        if any("jax" in x.__name__ for x in backend):
            return jax_lambdify()

    return sympy.lambdify(
        variables,
        expression,
        modules=backend_modules,
    )


class Intensity(Function):
    """Implementation of the `~.Function` from a sympy based model.

    For fast evaluations the sympy model is converted into a callable python
    function via `~sympy.utilities.lambdify.lambdify`, with many possible
    evaluation backends available.

    Args:
        model: A `.SympyModel` instance.
        backend: A string, tuple or mapping passed to the
          `~sympy.utilities.lambdify.lambdify` call as the :code:`modules`
          argument.

    """

    def __init__(
        self, model: SympyModel, backend: Union[str, tuple, dict] = "numpy"
    ):
        full_sympy_model = model.expression.doit()
        self.__input_variable_order = tuple(
            x.name for x in full_sympy_model.free_symbols
        )
        self.__callable_model = lambdify(
            tuple(full_sympy_model.free_symbols),
            full_sympy_model,
            backend=backend,
        )

        self.__parameters: Dict[str, Union[float, complex]] = {
            k.name: v for k, v in model.parameters.items()
        }

    def __call__(self, dataset: Dict[str, Any]) -> Any:
        """Evaluate the Intensity.

        Args:
            dataset: Contains all required kinematic variables.

        Returns:
            List of intensity values.

        """
        return self.__callable_model(
            *(
                dataset[var_name]
                if var_name in dataset
                else self.__parameters[var_name]
                for var_name in self.__input_variable_order
            )
        )

    @property
    def parameters(self) -> Dict[str, Union[float, complex]]:
        return self.__parameters

    def update_parameters(
        self, new_parameters: Dict[str, Union[float, complex]]
    ) -> None:
        for name, value in new_parameters.items():
            if name in self.__parameters:
                self.__parameters[name] = value
            else:
                logging.warning(
                    f"Updating the intensity with a parameter {name} which is "
                    f"not defined in the model!"
                )
