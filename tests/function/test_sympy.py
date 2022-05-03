# cspell:ignore lambdifygenerated
# pylint: disable=redefined-outer-name
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
    from _pytest.logging import LogCaptureFixture

__symbols: tuple[sp.Symbol, ...] = sp.symbols("a b c d x y z")
a, b, c, d, x, y, z = __symbols


def create_expression(a, x, y, z) -> sp.Expr:
    return a * (x**z + 2 * y)


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
def test_collect_constant_sub_expressions(free_symbols, expected):
    expression = a * x + b * (c * x**2 + d * x**2)
    sub_expresions = _collect_constant_sub_expressions(
        expression, free_symbols
    )
    assert sub_expresions == expected


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
def test_extract_constant_sub_expressions(free_symbols, expected_top):
    original_expression = a * x + b * (c * x**2 + d * x**2)
    top_expression, sub_exprs = extract_constant_sub_expressions(
        original_expression, free_symbols, fix_order=True
    )
    assert original_expression == top_expression.xreplace(sub_exprs)
    assert str(top_expression) == expected_top


def test_extract_constant_sub_expressions_warning(caplog: LogCaptureFixture):
    caplog.set_level(logging.INFO)
    expression = a * z**2

    caplog.clear()
    extract_constant_sub_expressions(expression, free_symbols=[c])
    assert "Symbol c does not appear in the expression" in caplog.text

    caplog.clear()
    extract_constant_sub_expressions(expression, free_symbols=[c, d])
    assert "Symbols c, d do not appear in the expression" in caplog.text


@pytest.mark.parametrize("backend", ["jax", "math", "numpy", "tf"])
def test_create_function(backend: str):
    expression = create_expression(a, x, y, z)
    function = create_function(expression, backend)
    assert callable(function.function)
    assert function.argument_order == ("a", "x", "y", "z")


@pytest.mark.parametrize("backend", ["jax", "math", "numpy", "tf"])
def test_create_function_indexed_symbol(backend: str):
    a = sp.IndexedBase("A")
    expr = a[0] ** 2 + a[1] ** 2
    func = create_function(expr, backend=backend)
    assert func.argument_order == ("A[0]", "A[1]")


@pytest.mark.parametrize("backend", ["jax", "math", "numpy", "tf"])
@pytest.mark.parametrize("max_complexity", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("use_cse", [False, True])
def test_fast_lambdify(backend: str, max_complexity: int, use_cse: bool):
    expression = create_expression(a, x, y, z)
    function = fast_lambdify(
        expression,
        symbols=(a, x, y, z),
        backend=backend,
        use_cse=use_cse,
        max_complexity=max_complexity,
    )

    func_repr = str(function)
    if 0 < max_complexity <= 4:
        repr_start = "<function fast_lambdify.<locals>"
    else:
        repr_start = "<function _lambdifygenerated"
    if backend == "jax":
        repr_start = "<CompiledFunction of " + repr_start
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


def test_prepare_caching():
    cache_expression, transformer_expressions = prepare_caching(
        expression=a * x + b * (c * x**2 + d * y**2),
        parameters={a: -2.5, b: 1, c: 0.0, d: 3.7},
        free_parameters={a, d},
    )
    f0 = sp.Symbol("f0")
    assert cache_expression == a * x + d * f0
    assert transformer_expressions == {x: x, f0: y**2}


def test_split_expression():
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

    free_symbols: set[sp.Symbol] = top_expr.free_symbols  # type: ignore[assignment]
    sub_symbols = sorted(free_symbols, key=lambda s: s.name)
    assert len(sub_symbols) == 3
    f0, f1, f2 = tuple(sub_symbols)
    assert f0 is a
    assert sub_expressions[f0] == a
    assert sub_expressions[f1] == x**z
    assert sub_expressions[f2] == 2 * y
