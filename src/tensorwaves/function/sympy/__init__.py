# pylint: disable=import-outside-toplevel, line-too-long
"""Lambdify `sympy` expression trees to a `.Function`."""
from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    Mapping,
    Sequence,
)

from tqdm.auto import tqdm

from tensorwaves.function import (
    ParametrizedBackendFunction,
    PositionalArgumentFunction,
)
from tensorwaves.function._backend import (
    get_backend_modules,
    jit_compile,
    raise_missing_module_error,
)
from tensorwaves.interface import ParameterValue

if TYPE_CHECKING:  # pragma: no cover
    import sympy as sp
    from sympy.printing.printer import Printer


def create_function(
    expression: sp.Expr,
    backend: str,
    use_cse: bool = True,
    max_complexity: int | None = None,
) -> PositionalArgumentFunction:
    """Convert a SymPy expression to a computational function.

    Args:
      expression: The SymPy expression that you want to
        `~sympy.utilities.lambdify.lambdify`. Its
        `~sympy.core.basic.Basic.free_symbols` become arguments to the
        resulting `.PositionalArgumentFunction`.

      backend: The computational backend in which to express the function.
      use_cse: Identify common sub-expressions in the function. This usually
        makes the function faster and speeds up lambdification.

      max_complexity: See :ref:`usage/faster-lambdify:Specifying complexity`
        and :doc:`compwa-org:report/002`.

    Example:
      >>> import numpy as np
      >>> import sympy as sp
      >>> from tensorwaves.function.sympy import create_function
      >>> x, y = sp.symbols("x y")
      >>> expression = x**2 + y**2
      >>> function = create_function(expression, backend="jax")
      >>> array = np.linspace(0, 3, num=4)
      >>> data = {"x": array, "y": array}
      >>> function(data)
      DeviceArray([  0.,  2.,  8., 18.], dtype=float64)
    """
    free_symbols = _get_free_symbols(expression)
    sorted_symbols = sorted(free_symbols, key=lambda s: s.name)
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
    expression: sp.Expr,
    parameters: Mapping[sp.Symbol, ParameterValue],
    backend: str,
    use_cse: bool = True,
    max_complexity: int | None = None,
) -> ParametrizedBackendFunction:
    """Convert a SymPy expression to a parametrized function.

    This is an extended version of :func:`create_function`, which allows one to
    identify certain symbols in the expression as parameters.

    Args:
      expression: See :func:`create_function`.
      parameters: The symbols in the expression that are be identified as
        `~.ParametrizedFunction.parameters` in the returned
        `.ParametrizedBackendFunction`.
      backend: See :func:`create_function`.
      use_cse: See :func:`create_function`.
      max_complexity: See :func:`create_function`.

    Example:
      >>> import numpy as np
      >>> import sympy as sp
      >>> from tensorwaves.function.sympy import create_parametrized_function
      >>> a, b, x, y = sp.symbols("a b x y")
      >>> expression = a * x**2 + b * y**2
      >>> function = create_parametrized_function(
      ...     expression,
      ...     parameters={a: -1, b: 2.5},
      ...     backend="jax",
      ... )
      >>> array = np.linspace(0, 1, num=5)
      >>> data = {"x": array, "y": array}
      >>> function.update_parameters({"b": 1})
      >>> function(data)
      DeviceArray([0., 0., 0., 0., 0.], dtype=float64)
    """
    free_symbols = _get_free_symbols(expression)
    sorted_symbols = sorted(free_symbols, key=lambda s: s.name)
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


def _get_free_symbols(expression: sp.Basic) -> set[sp.Symbol]:
    """Get free symbols in an expression, excluding IndexedBase.

    >>> import sympy as sp
    >>> A = sp.IndexedBase("A")
    >>> expr = A[0] ** 2 + A[1] ** 2
    >>> sorted(expr.free_symbols, key=str)
    [A, A[0], A[1]]
    >>> sorted(_get_free_symbols(expr), key=str)
    [A[0], A[1]]
    """
    import sympy as sp

    free_symbols: set[sp.Symbol] = expression.free_symbols  # type: ignore[assignment]
    index_bases = {
        sp.Symbol(s.base.name, **s.assumptions0)
        for s in free_symbols
        if isinstance(s, sp.Indexed)
    }
    return free_symbols - index_bases


def _lambdify_normal_or_fast(
    expression: sp.Expr,
    symbols: Sequence[sp.Symbol],
    backend: str,
    max_complexity: int | None,
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
    expression: sp.Expr,
    symbols: Sequence[sp.Symbol],
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
        try:
            # pylint: disable=import-error
            # pyright: reportMissingImports=false
            import tensorflow.experimental.numpy as tnp
        except ImportError:  # pragma: no cover
            raise_missing_module_error("tensorflow", extras_require="tf")
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
    expression: sp.Expr,
    symbols: Sequence[sp.Symbol],
    modules: str | tuple | dict,
    use_cse: bool,
    printer: Printer | None = None,
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
    expression: sp.Expr,
    symbols: Sequence[sp.Symbol],
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
    sub_functions: list[Callable] = []
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


def _collect_constant_sub_expressions(
    expression: sp.Basic, free_symbols: Iterable[sp.Symbol]
) -> set[sp.Expr]:
    import sympy as sp

    free_symbol_set = set(free_symbols)
    if not free_symbol_set:
        return set()

    def iterate_constant_sub_expressions(
        expression: sp.Basic,
    ) -> Generator[sp.Expr, None, None]:
        if isinstance(expression, sp.Atom):
            return
        if _get_free_symbols(expression) & free_symbol_set:
            for expr in expression.args:
                yield from iterate_constant_sub_expressions(expr)
            return
        yield expression  # type: ignore[misc]

    return set(iterate_constant_sub_expressions(expression))


def extract_constant_sub_expressions(
    expression: sp.Expr,
    free_symbols: Iterable[sp.Symbol],
    fix_order: bool = False,
) -> tuple[sp.Expr, dict[sp.Symbol, sp.Expr]]:
    """Collapse and extract constant sub-expressions.

    Along with :func:`prepare_caching`, this function prepares a `sympy.Expr
    <sympy.core.expr.Expr>` for caching during a fit procedure. The function
    returns a top expression where the constant sub-expressions have been
    substituted by new symbols :math:`f_i` for each substituted sub-expression,
    and a `dict` that gives the sub-expressions that those symbols represent.
    The top expression can be given to :func:`create_parametrized_function`,
    while the `dict` of sub-expressions can be given to a
    `.SympyDataTransformer.from_sympy`.

    Args:
        expression: The `~sympy.core.expr.Expr` from which to extract constant
            sub-expressions.
        free_symbols: `~sympy.core.symbol.Symbol` instance in the main
            :code:`expression` that are not constant.
        fix_order: If `False`, the generated symbols for the sub-expressions
            are not deterministic, because they depend on the hashes of those
            sub-expressions. Setting this to `True` makes the order
            deterministic, but this is slower, because requires lambdifying
            each sub-expression to `str` first.

    .. seealso:: :ref:`usage/caching:Extract constant sub-expressions`
    """
    import sympy as sp

    free_symbols = set(free_symbols)
    over_defined = free_symbols - _get_free_symbols(expression)
    if over_defined:
        over_defined_symbols = sorted(over_defined, key=str)
        symbol_names = ", ".join(map(str, over_defined_symbols))
        if len(over_defined) == 1:
            text = f"Symbol {symbol_names} does"
        else:
            text = f"Symbols {symbol_names} do"
        logging.warning(f"{text} not appear in the expression")

    constant_sub_expressions = list(
        _collect_constant_sub_expressions(expression, free_symbols)
    )
    if fix_order:
        constant_sub_expressions = sorted(constant_sub_expressions, key=str)
    substitutions = {
        expr: sp.Symbol(f"f{i}")
        for i, expr in enumerate(constant_sub_expressions)
    }
    top_expression: sp.Expr = expression.xreplace(substitutions)
    sub_expressions = {
        symbol: expr
        for expr, symbol in substitutions.items()
        if symbol in _get_free_symbols(top_expression)
    }
    return top_expression, sub_expressions


def prepare_caching(
    expression: sp.Expr,
    parameters: Mapping[sp.Symbol, ParameterValue],
    free_parameters: Iterable[sp.Symbol],
    fix_order: bool = False,
) -> tuple[sp.Expr, dict[sp.Symbol, sp.Expr]]:
    """Prepare an expression for optimizing with caching.

    When fitting a `.ParametrizedFunction`, only its free
    `.ParametrizedFunction.parameters` are updated on each iteration. This
    allows for an optimization: all sub-expressions that are unaffected by
    these free parameters can be cached as a constant `.DataSample`. The
    strategy here is to create a top expression that contains only the
    parameters that are to be optimized.

    Along with :func:`extract_constant_sub_expressions`, this function prepares
    a `sympy.Expr <sympy.core.expr.Expr>` for this caching procedure. The
    function returns a top expression where the constant sub-expressions have
    been substituted by new symbols :math:`f_i` for each substituted
    sub-expression and a `dict` that gives the sub-expressions that those
    symbols represent.

    The top expression can be given to :func:`create_parametrized_function`,
    while the `dict` of sub-expressions can be given to a
    `.SympyDataTransformer.from_sympy`.

    Args:
        expression: The `~sympy.core.expr.Expr` from which to extract constant
            sub-expressions.
        parameters: A mapping of values for each of the parameter symbols in
            the :code:`expression`. Parameters that are not
            :code:`free_parameters` are substituted in the returned expressions
            with :meth:`~sympy.core.basic.Basic.xreplace`.
        free_parameters: `~sympy.core.symbol.Symbol` instances in the main
            :code:`expression` that are to be considered parameters and that
            will be optimized by an `.Optimizer` later on.
        fix_order: If `False`, the generated symbols for the sub-expressions
            are not deterministic, because they depend on the hashes of those
            sub-expressions. Setting this to `True` makes the order
            deterministic, but this is slower, because requires lambdifying
            each sub-expression to `str` first.

    .. seealso:: :ref:`usage/caching:Extract constant sub-expressions`
    """
    free_parameter_values = {}
    fixed_parameter_values = {}
    for par, value in parameters.items():
        if par in free_parameters:
            free_parameter_values[par] = value
        else:
            fixed_parameter_values[par] = value
    expression = expression.xreplace(fixed_parameter_values)

    cache_expression, sub_expressions = extract_constant_sub_expressions(
        expression, free_parameters, fix_order
    )
    transformer_expressions = {}
    undefined_variables: set[sp.Symbol] = set()
    variables = _get_free_symbols(expression) - set(parameters)
    for symbol, sub_expr in sub_expressions.items():
        transformer_expressions[symbol] = sub_expr
        undefined_variables.update(variables - _get_free_symbols(sub_expr))
    for symbol in undefined_variables:
        transformer_expressions[symbol] = symbol
    return cache_expression, transformer_expressions


def split_expression(
    expression: sp.Expr,
    max_complexity: int,
    min_complexity: int = 1,
) -> tuple[sp.Expr, dict[sp.Symbol, sp.Expr]]:
    """Split an expression into a 'top expression' and several sub-expressions.

    Replace nodes in the expression tree of a `sympy.Expr
    <sympy.core.expr.Expr>` that lie within a certain complexity range (see
    :meth:`~sympy.core.basic.Basic.count_ops`) with symbols and keep a mapping
    of each to these symbols to the sub-expressions that they replaced.

    .. seealso:: :doc:`/usage/faster-lambdify`
    """
    import sympy as sp

    i = 0
    symbol_mapping: dict[sp.Symbol, sp.Expr] = {}
    n_operations = sp.count_ops(expression)
    if max_complexity <= 0 or n_operations < max_complexity:
        return expression, symbol_mapping
    progress_bar = tqdm(
        total=n_operations,
        desc="Splitting expression",
        unit="node",
        disable=not _use_progress_bar(),
    )

    def recursive_split(sub_expression: sp.Basic) -> sp.Expr:
        nonlocal i
        for arg in sub_expression.args:
            complexity = sp.count_ops(arg)
            if min_complexity <= complexity <= max_complexity:
                progress_bar.update(n=complexity)
                symbol = sp.Symbol(f"f{i}")
                i += 1
                symbol_mapping[symbol] = arg  # type: ignore[assignment]
                sub_expression = sub_expression.xreplace({arg: symbol})
            else:
                new_arg = recursive_split(arg)
                sub_expression = sub_expression.xreplace({arg: new_arg})
        return sub_expression  # type: ignore[return-value]

    top_expression = recursive_split(expression)
    free_symbols = _get_free_symbols(top_expression)
    remaining_symbols = free_symbols - set(symbol_mapping)
    symbol_mapping.update({s: s for s in remaining_symbols})
    remainder = progress_bar.total - progress_bar.n
    progress_bar.update(n=remainder)  # pylint crashes if total is set directly
    progress_bar.close()
    return top_expression, symbol_mapping


def _use_progress_bar() -> bool:
    return logging.getLogger().level <= logging.WARNING
