"""`~.Function` Adapter for `sympy` based models."""

from typing import Any, Dict, Union

import attr
import sympy

from tensorwaves.interfaces import Function


@attr.s(frozen=True)
class SympyModel:
    expression: sympy.Expr = attr.ib()
    parameters: Dict[sympy.Symbol, float] = attr.ib()
    variables: Dict[sympy.Symbol, sympy.Expr] = attr.ib()


class Intensity(Function):
    """Implementation of the `~.Function` from a sympy based model.

    For fast evaluations the sympy model is converted into a callable python
    function via `~sympy.utilities.lambdify.lambdify`, with many possible
    evaluation backends available.

    Args:
        model: A `~expertsystem.amplitude.sympy.ModelInfo` instance created
          via the `expertsystem`.
        backend: A string or mapping passed to the
          `~sympy.utilities.lambdify.lambdify` call as the :code:`modules`
          argument.

    """

    def __init__(self, model: SympyModel, backend: Union[str, dict] = "numpy"):
        full_sympy_model = model.expression.doit()
        self.__input_variable_order = tuple(
            x.name for x in full_sympy_model.free_symbols
        )
        self.__callable_model = sympy.lambdify(
            tuple(full_sympy_model.free_symbols),
            full_sympy_model,
            modules=backend,
        )

        self.__parameters: Dict[str, float] = {
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
    def parameters(self) -> Dict[str, float]:
        return self.__parameters

    def update_parameters(self, new_parameters: dict) -> None:
        for name, value in new_parameters.items():
            if name in self.__parameters:
                self.__parameters[name] = value
