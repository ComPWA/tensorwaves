"""Implementations of `.DataTransformer`."""

from typing import Dict, Mapping, Optional, Set

import sympy as sp

from tensorwaves.function import PositionalArgumentFunction
from tensorwaves.function.sympy import _lambdify_normal_or_fast
from tensorwaves.interface import DataSample, DataTransformer, Function


class SympyDataTransformer(DataTransformer):
    """Implementation of a `.DataTransformer`."""

    def __init__(self, functions: Mapping[str, Function]) -> None:
        if any(map(lambda f: not isinstance(f, Function), functions.values())):
            raise TypeError(
                "Not all values in the mapping are an instance of"
                f" {Function.__name__}"
            )
        self.__functions = dict(functions)

    @property
    def functions(self) -> Dict[str, Function]:
        """Read-only access to the internal mapping of functions."""
        return dict(self.__functions)

    def __call__(self, dataset: DataSample) -> DataSample:
        """Transform one `.DataSample` into another `.DataSample`."""
        return {
            key: function(dataset)
            for key, function in self.__functions.items()
        }

    @classmethod
    def from_sympy(
        cls,
        expressions: Dict[sp.Symbol, sp.Expr],
        backend: str,
        *,
        max_complexity: Optional[int] = None,
    ) -> "SympyDataTransformer":
        expanded_expressions: Dict[str, sp.Expr] = {
            k.name: expr.doit() for k, expr in expressions.items()
        }
        free_symbols: Set[sp.Symbol] = set()
        for expr in expanded_expressions.values():
            free_symbols |= expr.free_symbols
        ordered_symbols = tuple(sorted(free_symbols, key=lambda s: s.name))
        argument_order = tuple(map(str, ordered_symbols))
        functions = {}
        for variable_name, expr in expanded_expressions.items():
            function = _lambdify_normal_or_fast(
                expr, ordered_symbols, backend, max_complexity=max_complexity
            )
            functions[variable_name] = PositionalArgumentFunction(
                function, argument_order
            )
        return cls(functions)
