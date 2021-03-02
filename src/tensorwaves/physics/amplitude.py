"""`~.Function` Adapter for `sympy` based models."""

import logging
from typing import Any, Callable, Dict, Tuple, Union

import attr
import sympy

from tensorwaves.interfaces import Function


@attr.s(frozen=True)
class SympyModel:
    r"""Full definition of an arbitrary model based on sympy.

    Note that input for particle physics amplitude models are based on four
    momenta. However for reasons of convenience some models may define and use
    a distinct set of kinematic variables (e.g. in the helicity formalism:
    angles :math:`\\theta` and :math:`\\phi`). In this case a
    `~.interfaces.Kinematics` instance (Adapter) is needed to convert four
    momentum information into the custom set of kinematic variables.

    Args:
        expression : A sympy expression that contains the complete information
          of the model based on some inputs. The inputs are defined via the
          :code:`free_symbols` attribute of the sympy expression.
        parameters: Defines which inputs of the model are parameters. The keys
          represent the parameter set, while the values represent their default
          values. Consequently the variables of the model are defined as the
          intersection of the total input set with the parameter set.
    """

    expression: sympy.Expr = attr.ib()
    parameters: Dict[sympy.Symbol, Union[float, complex]] = attr.ib()


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


def lambdify(
    variables: Tuple[sympy.Symbol, ...],
    expression: sympy.Expr,
    backend: Union[str, tuple, dict],
) -> Callable:
    """Wrapper around `~sympy.utilities.lambdify.lambdify`.

    Unifies and simplifies the lambdification process to various backends.
    """
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
            # pylint: disable=import-error
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
        model: Complete model description, which can be initialized from
          a `~expertsystem.amplitude.helicity.HelicityModel`.
        backend: Choice of backend for fast evaluations. Argument is passed to
          the `~.lambdify` function.
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
