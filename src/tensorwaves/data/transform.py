"""Implementations of `.DataTransformer`."""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from tensorwaves.function import PositionalArgumentFunction
from tensorwaves.function.sympy import (
    _get_free_symbols,
    _lambdify_normal_or_fast,
)
from tensorwaves.interface import DataSample, DataTransformer, Function

if TYPE_CHECKING:  # pragma: no cover
    import sympy as sp


class IdentityTransformer(DataTransformer):
    """`.DataTransformer` that leaves a `.DataSample` intact."""

    def __call__(self, data: DataSample) -> DataSample:
        return data


class SympyDataTransformer(DataTransformer):
    """Implementation of a `.DataTransformer`."""

    def __init__(self, functions: Mapping[str, Function]) -> None:
        if any(not isinstance(f, Function) for f in functions.values()):
            raise TypeError(
                "Not all values in the mapping are an instance of"
                f" {Function.__name__}"
            )
        self.__functions = dict(functions)

    @property
    def functions(self) -> dict[str, Function]:
        """Read-only access to the internal mapping of functions."""
        return dict(self.__functions)

    def __call__(self, data: DataSample) -> DataSample:
        """Transform one `.DataSample` into another `.DataSample`."""
        return {
            key: function(data) for key, function in self.__functions.items()
        }

    @classmethod
    def from_sympy(
        cls,
        expressions: dict[sp.Symbol, sp.Expr],
        backend: str,
        *,
        use_cse: bool = True,
        max_complexity: int | None = None,
    ) -> SympyDataTransformer:
        expanded_expressions: dict[str, sp.Expr] = {
            k.name: expr.doit() for k, expr in expressions.items()
        }
        free_symbols: set[sp.Symbol] = set()
        for expr in expanded_expressions.values():
            free_symbols |= _get_free_symbols(expr)
        ordered_symbols = tuple(sorted(free_symbols, key=lambda s: s.name))
        argument_order = tuple(map(str, ordered_symbols))
        functions = {}
        for variable_name, expr in expanded_expressions.items():
            function = _lambdify_normal_or_fast(
                expr,
                ordered_symbols,
                backend,
                use_cse=use_cse,
                max_complexity=max_complexity,
            )
            functions[variable_name] = PositionalArgumentFunction(
                function, argument_order
            )
        return cls(functions)
