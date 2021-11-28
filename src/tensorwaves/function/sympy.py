"""Lambdify `sympy` expression trees from a `.Model` to a `.Function`."""

import logging
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import sympy as sp
from sympy.printing.numpy import (
    NumPyPrinter,
    _numpy_known_constants,
    _numpy_known_functions,
)
from tqdm.auto import tqdm

from tensorwaves._backend import get_backend_modules
from tensorwaves.interface import Model, ParameterValue

_jax_known_functions = {
    k: v.replace("numpy.", "jnp.") for k, v in _numpy_known_functions.items()
}
_jax_known_constants = {
    k: v.replace("numpy.", "jnp.") for k, v in _numpy_known_constants.items()
}


class _JaxPrinter(NumPyPrinter):  # pylint: disable=abstract-method
    # pylint: disable=invalid-name
    module_imports = {"jax": {"numpy as jnp"}}
    _module = "jnp"
    _kc = _jax_known_constants
    _kf = _jax_known_functions

    def _print_ComplexSqrt(self, expr: sp.Expr) -> str:  # noqa: N802
        x = self._print(expr.args[0])
        return (
            "jnp.select("
            f"[jnp.less({x}, 0), True], "
            f"[1j * jnp.sqrt(-{x}), jnp.sqrt({x})], "
            "default=jnp.nan)"
        )


def split_expression(
    expression: sp.Expr,
    max_complexity: int,
    min_complexity: int = 0,
) -> Tuple[sp.Expr, Dict[sp.Symbol, sp.Expr]]:
    """Split an expression into a 'top expression' and several sub-expressions.

    Replace nodes in the expression tree of a `sympy.Expr
    <sympy.core.expr.Expr>` that lie within a certain complexity range (see
    :meth:`~sympy.core.basic.Basic.count_ops`) with symbols and keep a mapping
    of each to these symbols to the sub-expressions that they replaced.

    .. seealso:: :doc:`/usage/faster-lambdify`
    """
    i = 0
    symbol_mapping: Dict[sp.Symbol, sp.Expr] = {}
    n_operations = sp.count_ops(expression)
    if n_operations < max_complexity:
        return expression, symbol_mapping
    progress_bar = tqdm(
        total=n_operations,
        desc="Splitting expression",
        unit="node",
        disable=not _use_progress_bar(),
    )

    def recursive_split(sub_expression: sp.Expr) -> sp.Expr:
        nonlocal i
        for arg in sub_expression.args:
            complexity = sp.count_ops(arg)
            if min_complexity < complexity < max_complexity:
                progress_bar.update(n=complexity)
                symbol = sp.Symbol(f"f{i}")
                i += 1
                symbol_mapping[symbol] = arg
                sub_expression = sub_expression.xreplace({arg: symbol})
            else:
                new_arg = recursive_split(arg)
                sub_expression = sub_expression.xreplace({arg: new_arg})
        return sub_expression

    top_expression = recursive_split(expression)
    remainder = progress_bar.total - progress_bar.n
    progress_bar.update(n=remainder)  # pylint crashes if total is set directly
    progress_bar.close()
    return top_expression, symbol_mapping


def fast_lambdify(
    expression: sp.Expr,
    symbols: Sequence[sp.Symbol],
    backend: Union[str, tuple, dict],
    *,
    min_complexity: int = 0,
    max_complexity: int,
    **kwargs: Any,
) -> Callable:
    """Speed up :func:`.lambdify` with :func:`.split_expression`.

    .. seealso:: :doc:`/usage/faster-lambdify`
    """
    top_expression, sub_expressions = split_expression(
        expression,
        min_complexity=min_complexity,
        max_complexity=max_complexity,
    )
    if not sub_expressions:
        return lambdify(top_expression, symbols, backend, **kwargs)

    sorted_top_symbols = sorted(sub_expressions, key=lambda s: s.name)
    top_function = lambdify(
        top_expression, sorted_top_symbols, backend, **kwargs
    )
    sub_functions: List[Callable] = []
    for symbol in tqdm(
        iterable=sorted_top_symbols,
        desc="Lambdifying sub-expressions",
        unit="expr",
        disable=not _use_progress_bar(),
    ):
        sub_expression = sub_expressions[symbol]
        sub_function = lambdify(sub_expression, symbols, backend, **kwargs)
        sub_functions.append(sub_function)

    def recombined_function(*args: Any) -> Any:
        new_args = [sub_function(*args) for sub_function in sub_functions]
        return top_function(*new_args)

    return recombined_function


def _sympy_lambdify(
    expression: sp.Expr,
    symbols: Sequence[sp.Symbol],
    backend: Union[str, tuple, dict],
    *,
    max_complexity: Optional[int] = None,
    **kwargs: Any,
) -> Callable:
    if max_complexity is None:
        return lambdify(
            expression=expression,
            symbols=symbols,
            backend=backend,
            **kwargs,
        )
    return fast_lambdify(
        expression=expression,
        symbols=symbols,
        backend=backend,
        max_complexity=max_complexity,
        **kwargs,
    )


def lambdify(
    expression: sp.Expr,
    symbols: Sequence[sp.Symbol],
    backend: Union[str, tuple, dict],
    **kwargs: Any,
) -> Callable:
    """A wrapper around :func:`~sympy.utilities.lambdify.lambdify`."""
    # pylint: disable=import-outside-toplevel,too-many-return-statements
    def jax_lambdify() -> Callable:
        import jax

        return jax.jit(
            sp.lambdify(
                symbols,
                expression,
                modules=modules,
                printer=_JaxPrinter,
                **kwargs,
            )
        )

    def numba_lambdify() -> Callable:
        # pylint: disable=import-error
        import numba

        return numba.jit(
            sp.lambdify(symbols, expression, modules="numpy", **kwargs),
            forceobj=True,
            parallel=True,
        )

    def tensorflow_lambdify() -> Callable:
        # pylint: disable=import-error
        import tensorflow.experimental.numpy as tnp  # pyright: reportMissingImports=false

        return sp.lambdify(symbols, expression, modules=tnp, **kwargs)

    modules = get_backend_modules(backend)
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
        if any(
            "tensorflow" in x.__name__ or "tf" in x.__name__ for x in backend
        ):
            return tensorflow_lambdify()

    return sp.lambdify(symbols, expression, modules=modules, **kwargs)


class SympyModel(Model):
    r"""Full definition of an arbitrary model based on `sympy`.

    Note that input for particle physics amplitude models are based on
    four-momenta. However, for reasons of convenience, some models may define
    and use a distinct set of kinematic variables (e.g. in the helicity
    formalism: angles :math:`\theta` and :math:`\phi`). In this case, a
    `.DataTransformer` instance (adapter) is needed to transform four momentum
    information into the custom set of kinematic variables.

    Args:
        expression: A sympy expression that contains the complete information
            of the model based on some inputs. The inputs are defined via the
            `~sympy.core.basic.Basic.free_symbols` attribute of the
            `sympy.Expr <sympy.core.expr.Expr>`.

        parameters: Defines which inputs of the model are parameters. The keys
            represent the parameter set, while the values represent their
            default values. Consequently, the variables of the model are
            defined as the intersection of the total input set with the
            parameter set.
    """

    def __init__(
        self,
        expression: sp.Expr,
        parameters: Dict[sp.Symbol, ParameterValue],
        max_complexity: Optional[int] = None,
    ) -> None:
        if not all(map(lambda p: isinstance(p, sp.Symbol), parameters)):
            raise TypeError(f"Not all parameters are of type {sp.Symbol}")

        if not set(parameters) <= set(expression.free_symbols):
            unused_parameters = set(parameters) - set(expression.free_symbols)
            logging.warning(
                f"Parameters {unused_parameters} are defined but do not appear"
                " in the model!"
            )

        self.__expression = expression
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
        self.max_complexity = max_complexity

    def lambdify(self, backend: Union[str, tuple, dict]) -> Callable:
        """Lambdify the model using `~sympy.utilities.lambdify.lambdify`."""
        return _sympy_lambdify(
            expression=self.__expression,
            symbols=self.__argument_order,
            backend=backend,
            max_complexity=self.max_complexity,
        )

    @property
    def parameters(self) -> Dict[str, ParameterValue]:
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


def _use_progress_bar() -> bool:
    return logging.getLogger().level <= logging.WARNING
