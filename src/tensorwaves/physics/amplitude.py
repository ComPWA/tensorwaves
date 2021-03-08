"""`~.Function` Adapter for `sympy` based models."""

from typing import Any, Callable, Dict, Union

import sympy

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

    def __init__(self, expression: sympy.Expr, parameters: Dict[sympy.Symbol, Union[float, complex]]) -> None:
        self.__expression = expression.doit()
        self.__parameters = parameters

    def lambdify(
        self,
        backend: Union[str, tuple, dict]
    ) -> Function:
        """Lambdify the model using `~sympy.utilities.lambdify.lambdify`."""
        # pylint: disable=import-outside-toplevel
        callable_model = None

        def function_wrapper(dataset: Dict[str, Any]) -> Any:
            return callable_model(
                *(
                    dataset[var_name]
                    if var_name in dataset
                    else self.__parameters[var_name]
                    for var_name in self.__input_variable_order
                )
            )

        self.__input_variable_order = tuple(
            x.name for x in self.__expression.free_symbols
        )

        backend_modules = get_backend_modules(backend)

        variables = tuple(self.__expression.free_symbols)

        def jax_lambdify() -> Callable:
            from jax import jit

            return jit(
                sympy.lambdify(
                    variables,
                    self.expression,
                    modules=backend_modules,
                )
            )

        if isinstance(backend, str):
            if backend == "jax":
                callable_model = jax_lambdify()
            if backend == "numba":
                # pylint: disable=import-error
                from numba import jit

                callable_model = jit(
                    sympy.lambdify(
                        variables,
                        self.__expression,
                        modules="numpy",
                    ),
                    parallel=True,
                )
        if isinstance(backend, tuple):
            if any("jax" in x.__name__ for x in backend):
                callable_model = jax_lambdify()

        callable_model = sympy.lambdify(
            variables,
            self.__expression,
            modules=backend_modules,
        )

        return function_wrapper

    @property
    def parameters(self) -> Dict[str, Union[float, complex]]:
        return self.__parameters

        self.__parameters: Dict[str, Union[float, complex]] = {
            k.name: v for k, v in model.parameters.items()
        }


class Intensity(Function):
    """Implementation of the `~.Function` from a sympy based model.

    For fast evaluations the sympy model is converted into a callable python
    function via `~sympy.utilities.lambdify.lambdify`, with many possible
    evaluation backends available.

    Args:
        model: Complete model description, which can be initialized from
          a `~expertsystem.amplitude.helicity.HelicityModel`.

    """

    def __init__(
        self, model: SympyModel, backend: Union[str, tuple, dict]
    ):






