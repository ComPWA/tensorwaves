from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import sympy as sp

from tensorwaves.function.sympy import (
    _collect_constant_sub_expressions,
    create_function,
    extract_constant_sub_expressions,
    fast_lambdify,
    prepare_caching,
    split_expression,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from _pytest.logging import LogCaptureFixture

a, b, c, d, x, y, z = cast("list[sp.Symbol]", sp.symbols("a b c d x y z"))


def create_expression(a, x, y, z) -> sp.Expr:
    return a * (x**z + 2 * y)


def describe_collect_constant_sub_expressions():
    @pytest.mark.parametrize(
        ("free_symbols", "expected"),
        [
            ([], set()),
            ([a], {b * (c * x**2 + d * x**2)}),
            ([b], {a * x, c * x**2 + d * x**2}),
            ([c], {a * x, x**2, d * x**2}),
            ([d], {a * x, c * x**2, x**2}),
            ([a, c, d], {x**2}),
            ([x], set()),
        ],
    )
    def it_collects_the_sub_expressions_without_free_symbols(free_symbols, expected):
        expression = a * x + b * (c * x**2 + d * x**2)
        sub_expresions = _collect_constant_sub_expressions(expression, free_symbols)
        assert sub_expresions == expected


def describe_create_function():
    @pytest.mark.parametrize("backend", ["jax", "math", "numpy", "tf"])
    def it_orders_the_arguments_as_the_symbols_appear(backend: str):
        expression = create_expression(a, x, y, z)
        function = create_function(expression, backend)
        assert callable(function.function)
        assert function.argument_order == ("a", "x", "y", "z")

    @pytest.mark.parametrize("backend", ["jax", "math", "numpy", "tf"])
    def it_names_indexed_symbols_by_their_index(backend: str):
        a = sp.IndexedBase("A")
        expr = a[0] ** 2 + a[1] ** 2
        func = create_function(expr, backend=backend)
        assert func.argument_order == ("A[0]", "A[1]")

    @pytest.mark.parametrize("backend", ["jax", "math", "numpy", "tf"])
    def it_names_matrix_symbols_by_their_element(backend: str):
        M = sp.MatrixSymbol("M", 2, 2)  # ruff:ignore[non-lowercase-variable-in-function]
        expr = M[0, 0] ** 2 + M[1, 1] ** 2
        func = create_function(expr, backend=backend)
        assert func.argument_order == ("M[0, 0]", "M[1, 1]")


def describe_extract_constant_sub_expressions():
    @pytest.mark.parametrize(
        ("free_symbols", "expected_top"),
        [
            ([], "a*x + b*(c*x**2 + d*x**2)"),
            ([a], "a*x + f0"),
            ([a, b], "a*x + b*f0"),
            ([a, c], "a*x + b*(c*f1 + f0)"),
            ([a, c, d], "a*x + b*(c*f0 + d*f0)"),
            ([a, x], "a*x + b*(c*x**2 + d*x**2)"),
            ([a, b, c, d, x], "a*x + b*(c*x**2 + d*x**2)"),
        ],
    )
    def it_substitutes_the_constant_sub_expressions_by_symbols(
        free_symbols, expected_top
    ):
        original_expression = a * x + b * (c * x**2 + d * x**2)
        top_expression, sub_exprs = extract_constant_sub_expressions(
            original_expression, free_symbols, fix_order=True
        )
        assert original_expression == top_expression.xreplace(sub_exprs)
        assert str(top_expression) == expected_top

    def it_warns_about_free_symbols_that_do_not_appear(caplog: LogCaptureFixture):
        caplog.set_level(logging.INFO)
        expression = a * z**2

        caplog.clear()
        extract_constant_sub_expressions(expression, free_symbols=[c])
        assert "Symbol c does not appear in the expression" in caplog.text

        caplog.clear()
        extract_constant_sub_expressions(expression, free_symbols=[c, d])
        assert "Symbols c, d do not appear in the expression" in caplog.text


def describe_fast_lambdify():
    @pytest.mark.parametrize("backend", ["jax", "math", "numpy", "tf"])
    @pytest.mark.parametrize("max_complexity", [0, 1, 2, 3, 4, 5])
    @pytest.mark.parametrize("use_cse", [False, True])
    @pytest.mark.parametrize("use_jit", [False, True, None])
    def it_evaluates_the_same_as_the_original_expression(
        backend: str, max_complexity: int, use_cse: bool, use_jit: bool | None
    ):
        def call_fast_lambdify() -> Callable:
            return fast_lambdify(
                expression=create_expression(a, x, y, z),
                symbols=(a, x, y, z),
                backend=backend,
                use_cse=use_cse,
                use_jit=use_jit,
                max_complexity=max_complexity,
            )

        if use_jit and backend not in {"jax", "numba"}:
            with pytest.warns(
                UserWarning,
                match=f"Backend {backend} does not yet() support JIT compilation",
            ):
                function = call_fast_lambdify()
        else:
            function = call_fast_lambdify()

        func_repr = str(function)
        if 0 < max_complexity <= 4:
            repr_start = "<function fast_lambdify.<locals>"
        else:
            # cspell:ignore lambdifygenerated
            repr_start = "<function _lambdifygenerated"
        if backend == "jax" and use_jit is not False:
            repr_start = "<PjitFunction of " + repr_start
            # cspell:ignore Pjit
        assert func_repr.startswith(repr_start)

        data = (
            4,
            np.array([1, 2]),
            np.array([1, np.e]),
            np.array([1, 2]),
        )
        output = function(*data)
        expected = create_expression(*data)
        assert pytest.approx(output) == expected


def describe_prepare_caching():
    def it_moves_the_constant_sub_expressions_into_the_transformer():
        cache_expression, transformer_expressions = prepare_caching(
            expression=a * x + b * (c * x**2 + d * y**2),
            parameters={a: -2.5, b: 1, c: 0.0, d: 3.7},
            free_parameters={a, d},
        )
        f0 = sp.Symbol("f0")
        assert cache_expression == a * x + d * f0
        assert transformer_expressions == {x: x, f0: y**2}


def describe_split_expression():
    def it_splits_off_sub_expressions_within_the_complexity_range():
        expression = create_expression(a, x, y, z)

        assert expression.args[0] is a
        assert len(expression.args[1].args) == 2
        sub_expr, _ = expression.args[1].args
        assert sub_expr == x**z
        n_nodes = sp.count_ops(sub_expr)
        assert n_nodes == 1

        top_expr, sub_expressions = split_expression(
            expression,
            min_complexity=n_nodes,
            max_complexity=n_nodes,
        )
        assert top_expr.free_symbols == set(sub_expressions)
        assert expression == top_expr.xreplace(sub_expressions)

        free_symbols = cast("set[sp.Symbol]", top_expr.free_symbols)
        sub_symbols = sorted(free_symbols, key=str)
        assert len(sub_symbols) == 3
        f0, f1, f2 = tuple(sub_symbols)
        assert f0 is a
        assert sub_expressions[f0] == a
        assert sub_expressions[f1] == x**z
        assert sub_expressions[f2] == 2 * y
