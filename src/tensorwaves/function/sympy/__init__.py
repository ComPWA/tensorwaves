# pylint: disable=import-outside-toplevel
"""Lambdify `sympy` expression trees to a `.Function`."""

import logging
from typing import (
    TYPE_CHECKING,
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

from tqdm.auto import tqdm

from tensorwaves.function import (
    ParametrizedBackendFunction,
    PositionalArgumentFunction,
)
from tensorwaves.function._backend import get_backend_modules, jit_compile
from tensorwaves.interface import ParameterValue

if TYPE_CHECKING:
    import sympy as sp
    from sympy.printing.printer import Printer


def create_function(
    expression: "sp.Expr",
    backend: str,
    max_complexity: Optional[int] = None,
    use_cse: bool = True,
) -> PositionalArgumentFunction:
    sorted_symbols = sorted(expression.free_symbols, key=lambda s: s.name)
    lambdified_function = _lambdify_normal_or_fast(
        expression=expression,
        symbols=sorted_symbols,
        backend=backend,
        max_complexity=max_complexity,
        use_cse=use_cse,
    )
    return PositionalArgumentFunction(
        function=lambdified_function,
        argument_order=tuple(map(str, sorted_symbols)),
    )


def create_parametrized_function(
    expression: "sp.Expr",
    parameters: Mapping["sp.Symbol", ParameterValue],
    backend: str,
    max_complexity: Optional[int] = None,
    use_cse: bool = True,
) -> ParametrizedBackendFunction:
    sorted_symbols = sorted(expression.free_symbols, key=lambda s: s.name)
    lambdified_function = _lambdify_normal_or_fast(
        expression=expression,
        symbols=sorted_symbols,
        backend=backend,
        max_complexity=max_complexity,
        use_cse=use_cse,
    )
    return ParametrizedBackendFunction(
        function=lambdified_function,
        argument_order=tuple(map(str, sorted_symbols)),
        parameters={
            symbol.name: value for symbol, value in parameters.items()
        },
    )


def _lambdify_normal_or_fast(
    expression: "sp.Expr",
    symbols: Sequence["sp.Symbol"],
    backend: str,
    max_complexity: Optional[int],
    use_cse: bool,
) -> Callable:
    """Switch between `.lambdify` and `.fast_lambdify`."""
    if max_complexity is None:
        return lambdify(
            expression=expression,
            symbols=symbols,
            backend=backend,
            use_cse=use_cse,
        )
    return fast_lambdify(
        expression=expression,
        symbols=symbols,
        backend=backend,
        max_complexity=max_complexity,
        use_cse=use_cse,
    )


def lambdify(
    expression: "sp.Expr",
    symbols: Sequence["sp.Symbol"],
    backend: str,
    use_cse: bool = True,
) -> Callable:
    """A wrapper around :func:`~sympy.utilities.lambdify.lambdify`.

    Args:
        expression: the `sympy.Expr <sympy.core.expr.Expr>` that you want to
            express as a function in a certain computation back-end.
        symbols: The `~sympy.core.symbol.Symbol` instances in the expression
            that you want to serve as **positional arguments** in the
            lambdified function. Note that positional arguments are
            **ordered**.
        backend: Computational back-end in which to express the lambdified
            function.
        use_cse: Lambdify with common sub-expressions (see :code:`cse` argument
            in :func:`~sympy.utilities.lambdify.lambdify`).
    """
    # pylint: disable=import-outside-toplevel, too-many-return-statements
    def jax_lambdify() -> Callable:
        from ._printer import JaxPrinter

        return jit_compile(backend="jax")(
            _sympy_lambdify(
                expression,
                symbols,
                modules=modules,
                printer=JaxPrinter(),
                use_cse=use_cse,
            )
        )

    def numba_lambdify() -> Callable:
        return jit_compile(backend="numba")(
            _sympy_lambdify(
                expression,
                symbols,
                use_cse=use_cse,
                modules="numpy",
            )
        )

    def tensorflow_lambdify() -> Callable:
        # pylint: disable=import-error
        # pyright: reportMissingImports=false
        import tensorflow.experimental.numpy as tnp

        from ._printer import TensorflowPrinter

        return _sympy_lambdify(
            expression,
            symbols,
            modules=tnp,
            printer=TensorflowPrinter(),
            use_cse=use_cse,
        )

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

    return _sympy_lambdify(
        expression,
        symbols,
        modules=modules,
        use_cse=use_cse,
    )


def _sympy_lambdify(
    expression: "sp.Expr",
    symbols: Sequence["sp.Symbol"],
    modules: Union[str, tuple, dict],
    use_cse: bool,
    printer: Optional["Printer"] = None,
) -> Callable:
    import sympy as sp

    if use_cse:
        dummy_replacements = {
            symbol: sp.Symbol(f"z{i}", **symbol.assumptions0)
            for i, symbol in enumerate(symbols)
        }
        expression = expression.xreplace(dummy_replacements)
        symbols = [dummy_replacements[s] for s in symbols]
    return sp.lambdify(
        symbols,
        expression,
        cse=use_cse,
        modules=modules,
        printer=printer,
    )


def fast_lambdify(  # pylint: disable=too-many-locals
    expression: "sp.Expr",
    symbols: Sequence["sp.Symbol"],
    backend: str,
    *,
    min_complexity: int = 0,
    max_complexity: int,
    use_cse: bool = True,
) -> Callable:
    """Speed up :func:`.lambdify` with :func:`.split_expression`.

    For a simple example of the reasoning behind this, see
    :doc:`/usage/faster-lambdify`.
    """
    top_expression, sub_expressions = split_expression(
        expression,
        min_complexity=min_complexity,
        max_complexity=max_complexity,
    )
    if not sub_expressions:
        return lambdify(top_expression, symbols, backend, use_cse=use_cse)

    sorted_top_symbols = sorted(sub_expressions, key=lambda s: s.name)
    top_function = lambdify(
        top_expression, sorted_top_symbols, backend, use_cse=use_cse
    )
    sub_functions: List[Callable] = []
    for symbol in tqdm(
        iterable=sorted_top_symbols,
        desc="Lambdifying sub-expressions",
        unit="expr",
        disable=not _use_progress_bar(),
    ):
        sub_expression = sub_expressions[symbol]
        sub_function = lambdify(
            sub_expression, symbols, backend, use_cse=use_cse
        )
        sub_functions.append(sub_function)

    @jit_compile(backend)  # type: ignore[arg-type]
    def recombined_function(*args: Any) -> Any:
        new_args = [sub_function(*args) for sub_function in sub_functions]
        return top_function(*new_args)

    return recombined_function


def split_expression(
    expression: "sp.Expr",
    max_complexity: int,
    min_complexity: int = 1,
) -> Tuple["sp.Expr", Dict["sp.Symbol", "sp.Expr"]]:
    """Split an expression into a 'top expression' and several sub-expressions.

    Replace nodes in the expression tree of a `sympy.Expr
    <sympy.core.expr.Expr>` that lie within a certain complexity range (see
    :meth:`~sympy.core.basic.Basic.count_ops`) with symbols and keep a mapping
    of each to these symbols to the sub-expressions that they replaced.

    .. seealso:: :doc:`/usage/faster-lambdify`
    """
    import sympy as sp

    i = 0
    symbol_mapping: Dict[sp.Symbol, sp.Expr] = {}
    n_operations = sp.count_ops(expression)
    if max_complexity <= 0 or n_operations < max_complexity:
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
            if min_complexity <= complexity <= max_complexity:
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
    remaining_symbols = top_expression.free_symbols - set(symbol_mapping)
    symbol_mapping.update({s: s for s in remaining_symbols})
    remainder = progress_bar.total - progress_bar.n
    progress_bar.update(n=remainder)  # pylint crashes if total is set directly
    progress_bar.close()
    return top_expression, symbol_mapping


def _use_progress_bar() -> bool:
    return logging.getLogger().level <= logging.WARNING
