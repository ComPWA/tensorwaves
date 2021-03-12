"""Evaluateable physics models for amplitude analysis.

The `.model` module takes care of lambdifying mathematical expressions to
computational backends. Currently, mathematical expressions are implemented
as `.sympy` expressions only.
"""

__all__ = [
    "sympy",
]

from . import sympy
