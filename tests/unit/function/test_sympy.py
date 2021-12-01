# cspell:ignore lambdifygenerated
import sys

import numpy as np
import pytest
import sympy as sp

from tensorwaves.function.sympy import fast_lambdify, split_expression


def create_expression(x, y, z):
    return x ** z + 2 * y


@pytest.mark.parametrize("backend", ["jax", "math", "numpy", "tf"])
@pytest.mark.parametrize("max_complexity", [0, 2, 3, 4])
def test_fast_lambdify(backend: str, max_complexity: int):
    x, y, z = sp.symbols("x y z")
    expression = create_expression(x, y, z)
    function = fast_lambdify(
        expression,
        symbols=[x, y, z],
        max_complexity=max_complexity,
        backend=backend,
    )

    func_repr = str(function)
    if max_complexity <= 3:
        assert func_repr.startswith("<function fast_lambdify.<locals>")
    else:
        repr_start = "<function _lambdifygenerated"
        if backend == "jax":
            if sys.version_info >= (3, 7):
                repr_start = "<CompiledFunction of " + repr_start
            else:
                repr_start = "<CompiledFunction object at 0x"
        assert func_repr.startswith(repr_start)

    data = (
        np.array([1, 2]),
        np.array([1, np.e]),
        np.array([1, 2]),
    )
    output = function(*data)
    expected = create_expression(*data)
    assert pytest.approx(output) == expected


def test_split_expression():
    x, y, z = sp.symbols("x y z")
    expression = create_expression(x, y, z)
    top_expr, sub_expressions = split_expression(expression, max_complexity=2)
    assert top_expr.free_symbols == set(sub_expressions)
    assert expression == top_expr.xreplace(sub_expressions)

    sub_symbols = sorted(top_expr.free_symbols, key=lambda s: s.name)
    assert len(sub_symbols) == 2
    f1, f2 = tuple(sub_symbols)
    assert sub_expressions[f1] == x ** z
    assert sub_expressions[f2] == 2 * y
