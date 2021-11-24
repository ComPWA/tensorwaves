"""Lambdify `sympy` expression trees to a `.Function`."""

import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
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
from tensorwaves.interface import ParameterValue

from . import ParametrizedBackendFunction

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


def create_function(
    expression: sp.Expr,
    parameters: Mapping[sp.Symbol, ParameterValue],
    backend: str,
    max_complexity: Optional[int] = None,
    **kwargs: Any,
) -> ParametrizedBackendFunction:
    sorted_symbols = sorted(expression.free_symbols, key=lambda s: s.name)
    if max_complexity is None:
        lambdified_function = lambdify(
            expression=expression,
            symbols=sorted_symbols,
            backend=backend,
            **kwargs,
        )
    else:
        lambdified_function = fast_lambdify(
            expression=expression,
            symbols=sorted_symbols,
            backend=backend,
            max_complexity=max_complexity,
            **kwargs,
        )
    return ParametrizedBackendFunction(
        function=lambdified_function,
        argument_order=list(map(str, sorted_symbols)),
        parameters={
            symbol.name: value for symbol, value in parameters.items()
        },
    )


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


def _use_progress_bar() -> bool:
    return logging.getLogger().level <= logging.WARNING
