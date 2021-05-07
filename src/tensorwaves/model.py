"""Evaluateable physics models for amplitude analysis.

The `.model` module takes care of lambdifying mathematical expressions to
computational backends. Currently, mathematical expressions are implemented
as `sympy` expressions only.
"""
# cspell: ignore xreplace
import copy
import logging
from typing import Any, Callable, Dict, FrozenSet, Mapping, Tuple, Union

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
        if backend in {"numpy", "numba"}:
            return (np, np.__dict__)
            # returning only np.__dict__ does not work well with conditionals
        if backend in {"tensorflow", "tf"}:
            # pylint: disable=import-error
            import tensorflow.experimental.numpy as tnp  # pyright: reportMissingImports=false

            return tnp.__dict__

    return backend


def _sympy_lambdify(
    ordered_symbols: Tuple[sp.Symbol, ...],
    expression: sp.Expr,
    backend: Union[str, tuple, dict],
) -> Callable:
    # pylint: disable=import-outside-toplevel,too-many-return-statements
    def jax_lambdify() -> Callable:
        import jax

        return jax.jit(
            sp.lambdify(
                ordered_symbols,
                expression,
                modules=backend_modules,
            )
        )

    def numba_lambdify() -> Callable:
        # pylint: disable=import-error
        import numba

        return numba.jit(
            sp.lambdify(
                ordered_symbols,
                expression,
                modules="numpy",
            ),
            forceobj=True,
            parallel=True,
        )

    def tensorflow_lambdify() -> Callable:
        # pylint: disable=import-error
        import tensorflow.experimental.numpy as tnp  # pyright: reportMissingImports=false

        return sp.lambdify(
            ordered_symbols,
            expression,
            modules=tnp,
        )

    backend_modules = get_backend_modules(backend)
    if isinstance(backend, str):
        if backend == "jax":
            return jax_lambdify()
        if backend == "numba":
            return numba_lambdify()
        if backend in {"tensorflow", "tf"}:
            return tensorflow_lambdify()
    if isinstance(backend, tuple):
        if any("jax" in x.__name__ for x in backend):
            return jax_lambdify()
        if any("numba" in x.__name__ for x in backend):
            return numba_lambdify()
        if any("tensorflow" in x.__name__ for x in backend) or any(
            "tf" in x.__name__ for x in backend
        ):
            return tensorflow_lambdify()
    return sp.lambdify(
        ordered_symbols,
        expression,
        modules=backend_modules,
    )


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


class _ConstantSubExpressionSympyModel(Model):
    """Implements a performance optimized sympy based model.

    Based on which symbols of the sympy expression are declared.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        expression: sp.Expr,
        parameters: Dict[sp.Symbol, Union[float, complex]],
        fix_inputs: DataSample,
    ) -> None:
        self.__fix_inputs = fix_inputs
        self.__constant_symbols = set(self.__fix_inputs)
        self.__constant_sub_expressions: Dict[sp.Symbol, sp.Expr] = {}
        self.__find_constant_subexpressions(expression)
        self.__expression = self.__replace_constant_sub_expressions(expression)
        self.__not_fixed_parameters = {
            k: v
            for k, v in parameters.items()
            if k.name not in self.__constant_symbols
        }
        self.__not_fixed_variables: FrozenSet[sp.Symbol] = frozenset(
            s
            for s in self.__expression.free_symbols
            if s.name not in self.parameters
            and s.name not in self.__constant_symbols
            and s not in self.__constant_sub_expressions
        )
        self.__argument_order = tuple(self.__not_fixed_variables) + tuple(
            self.__not_fixed_parameters
        )

    def __find_constant_subexpressions(self, expr: sp.Expr) -> bool:
        if not expr.args:
            if (
                isinstance(expr, sp.Symbol)
                and expr.name not in self.__constant_symbols
            ):
                return False
            return True

        is_constant = True
        temp_constant_sub_expression = []
        for arg in expr.args:
            if self.__find_constant_subexpressions(arg):
                if arg.args:
                    temp_constant_sub_expression.append(arg)
            else:
                is_constant = False

        if not is_constant:
            for sub_expr in temp_constant_sub_expression:
                placeholder = sp.Symbol(f"cached[{str(sub_expr)}]")
                self.__constant_sub_expressions[placeholder] = sub_expr
        return is_constant

    def __replace_constant_sub_expressions(
        self, expression: sp.Expr
    ) -> sp.Expr:
        new_expression = copy.deepcopy(expression)
        return new_expression.xreplace(
            {v: k for k, v in self.__constant_sub_expressions.items()}
        )

    def lambdify(self, backend: Union[str, tuple, dict]) -> Callable:
        input_symbols = tuple(self.__expression.free_symbols)
        lambdified_model = _sympy_lambdify(
            input_symbols,
            self.__expression,
            backend=backend,
        )
        constant_input_storage = {}
        for placeholder, sub_expr in self.__constant_sub_expressions.items():
            temp_lambdify = _sympy_lambdify(
                tuple(sub_expr.free_symbols), sub_expr, backend
            )
            free_symbol_names = {x.name for x in sub_expr.free_symbols}
            constant_input_storage[placeholder.name] = temp_lambdify(
                *(self.__fix_inputs[k] for k in free_symbol_names)
            )

        input_args: list = []
        non_fixed_arg_positions = list(range(0, len(self.argument_order)))

        for input_arg in input_symbols:
            if input_arg in self.__argument_order:
                non_fixed_arg_positions[
                    self.__argument_order.index(input_arg)
                ] = len(input_args)
                input_args.append(0.0)
            elif input_arg.name in self.__fix_inputs:
                input_args.append(self.__fix_inputs[input_arg.name])
            else:
                input_args.append(constant_input_storage[input_arg.name])

        def update_args(*args: Tuple[Any, ...]) -> None:
            for i, x in enumerate(args):
                input_args[non_fixed_arg_positions[i]] = x

        def wrapper(*args: Tuple[Any, ...]) -> Any:
            update_args(*args)
            return lambdified_model(*input_args)

        return wrapper

    def performance_optimize(self, fix_inputs: DataSample) -> "Model":
        return NotImplemented

    @property
    def parameters(self) -> Dict[str, Union[float, complex]]:
        return {
            symbol.name: value
            for symbol, value in self.__not_fixed_parameters.items()
        }

    @property
    def variables(self) -> FrozenSet[str]:
        """Expected input variable names."""
        return frozenset(
            {symbol.name for symbol in self.__not_fixed_variables}
        )

    @property
    def argument_order(self) -> Tuple[str, ...]:
        return tuple(x.name for x in self.__argument_order)


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
        if not all(map(lambda p: isinstance(p, sp.Symbol), parameters)):
            raise TypeError(f"Not all parameters are of type {sp.Symbol}")

        if not set(parameters) <= set(expression.free_symbols):
            unused_parameters = set(parameters) - set(expression.free_symbols)
            logging.warning(
                f"Parameters {unused_parameters} are defined but do not appear"
                " in the model!"
            )

        self.__expression: sp.Expr = expression.doit()
        # after .doit() certain symbols like the meson radius can disappear
        # hence the parameters have to be shrunk to this space
        self.__parameters = {
            k: v
            for k, v in parameters.items()
            if k in self.__expression.free_symbols
        }
        self.__variables: FrozenSet[sp.Symbol] = frozenset(
            s
            for s in self.__expression.free_symbols
            if s.name not in self.parameters
        )
        self.__argument_order = tuple(self.__variables) + tuple(
            self.__parameters
        )

    def lambdify(self, backend: Union[str, tuple, dict]) -> Callable:
        """Lambdify the model using `~sympy.utilities.lambdify.lambdify`."""
        return _sympy_lambdify(
            self.__argument_order, self.__expression, backend
        )

    def performance_optimize(self, fix_inputs: DataSample) -> "Model":
        return _ConstantSubExpressionSympyModel(
            self.__expression, self.__parameters, fix_inputs
        )

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
