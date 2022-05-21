"""Express mathematical expressions in terms of computational functions."""
from __future__ import annotations

import inspect
from typing import Callable, Iterable, Mapping

import attrs
import numpy as np
from attrs import field, frozen

from tensorwaves.interface import (
    DataSample,
    Function,
    ParameterValue,
    ParametrizedFunction,
)


def _all_str(
    _: PositionalArgumentFunction, __: attrs.Attribute, value: Iterable[str]
) -> None:
    if not all(isinstance(s, str) for s in value):
        raise TypeError(f"Not all arguments are of type {str.__name__}")


def _all_unique(
    _: PositionalArgumentFunction, __: attrs.Attribute, value: Iterable[str]
) -> None:
    argument_names = list(value)
    if len(set(argument_names)) != len(argument_names):
        duplicate_arguments = []
        for arg_name in argument_names:
            n_occurrences = argument_names.count(arg_name)
            if n_occurrences > 1:
                duplicate_arguments.append(arg_name)
        raise ValueError(
            "There are duplicate argument names:"
            f" {sorted(set(duplicate_arguments))}"
        )


def _validate_arguments(
    instance: PositionalArgumentFunction, _: attrs.Attribute, value: Callable
) -> None:
    if not callable(value):
        raise TypeError("Function is not callable")
    n_args = len(instance.argument_order)
    signature = inspect.signature(value)
    if len(signature.parameters) != n_args:
        if len(signature.parameters) == 1:
            parameter = next(iter(signature.parameters.values()))
            if parameter.kind == parameter.VAR_POSITIONAL:
                return
        raise ValueError(
            f"Lambdified function expects {len(signature.parameters)}"
            f" arguments, but {n_args} sorted arguments were provided."
        )


def _to_tuple(argument_order: Iterable[str]) -> tuple[str, ...]:
    return tuple(argument_order)


@frozen
class PositionalArgumentFunction(Function):
    """Wrapper around a function with positional arguments.

    This class provides a :meth:`~.Function.__call__` that can take a
    `.DataSample` for a function with `positional arguments
    <https://docs.python.org/3/glossary.html#term-positional-argument>`_. Its
    :attr:`argument_order` redirect the keys in the `.DataSample` to the
    argument positions in its underlying :attr:`function`.

    .. seealso:: :func:`.create_function`
    """

    function: Callable[..., np.ndarray] = field(validator=_validate_arguments)
    """A function with positional arguments only."""
    argument_order: tuple[str, ...] = field(
        converter=_to_tuple, validator=[_all_str, _all_unique]
    )
    """Ordered labels for each positional argument."""

    def __call__(self, data: DataSample) -> np.ndarray:
        args = [data[var_name] for var_name in self.argument_order]
        return self.function(*args)


class ParametrizedBackendFunction(ParametrizedFunction):
    """Implements `.ParametrizedFunction` for a specific computational back-end.

    .. seealso:: :func:`.create_parametrized_function`
    """

    def __init__(
        self,
        function: Callable[..., np.ndarray],
        argument_order: Iterable[str],
        parameters: Mapping[str, ParameterValue],
    ) -> None:
        self.__function = PositionalArgumentFunction(function, argument_order)
        self.__parameters = dict(parameters)

    def __call__(self, data: DataSample) -> np.ndarray:
        extended_data = {**data, **self.__parameters}  # type: ignore[arg-type]
        return self.__function(extended_data)

    @property
    def function(self) -> Callable[..., np.ndarray]:
        return self.__function.function

    @property
    def argument_order(self) -> tuple[str, ...]:
        return self.__function.argument_order

    @property
    def parameters(self) -> dict[str, ParameterValue]:
        return dict(self.__parameters)

    def update_parameters(
        self, new_parameters: Mapping[str, ParameterValue]
    ) -> None:
        over_defined = set(new_parameters) - set(self.__parameters)
        if over_defined:
            sep = "\n    "
            parameter_listing = f"{sep}".join(sorted(self.__parameters))
            raise ValueError(
                f"Parameters {over_defined} do not exist in function"
                f" arguments. Expecting one of:{sep}{parameter_listing}"
            )
        self.__parameters.update(new_parameters)


def get_source_code(function: Function) -> str:
    """Get the backend source code used to compile this function.

    >>> import sympy as sp
    >>> from tensorwaves.function.sympy import create_function
    >>> x, y = sp.symbols("x y")
    >>> expr = x**2 + y**2
    >>> func = create_function(expr, backend="jax", use_cse=False)
    >>> src = get_source_code(func)
    >>> print(src)
    def _lambdifygenerated(x, y):
        return x**2 + y**2
    """
    if isinstance(
        function, (PositionalArgumentFunction, ParametrizedBackendFunction)
    ):
        return inspect.getsource(function.function)
    raise NotImplementedError(
        f"Cannot get source code for {Function.__name__} type"
        f" {type(function).__name__}"
    )
