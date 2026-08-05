"""Express mathematical expressions in terms of computational functions."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import attrs
import numpy as np
from attrs import field, frozen

from tensorwaves.interface import (
    DataSample,
    Function,
    ParameterType,
    ParameterValue,
    ParametrizedFunction,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping


@runtime_checkable
class BackendFunction(Protocol):
    """A function object that exposes its backend-native kernel.

    Classes like `PositionalArgumentFunction` and `ParametrizedBackendFunction` wrap a
    pure function that takes positional argument arrays. This protocol gives access to
    that kernel and the backend it was compiled for, so that backend-native
    transformations (such as :code:`jax.jit`, :code:`jax.grad`, or :code:`jax.vmap`)
    can be applied to it and estimators can determine which computational backend to
    use.

    >>> import sympy as sp
    >>> from tensorwaves.function.sympy import create_function
    >>> x, y = sp.symbols("x y")
    >>> func = create_function(x**2 + y**2, backend="jax")
    >>> func.backend
    'jax'
    >>> func.argument_order
    ('x', 'y')
    """

    @property
    def function(self) -> Callable[..., np.ndarray]:
        """Backend-native function that takes positional arguments only."""

    @property
    def argument_order(self) -> tuple[str, ...]:
        """Name of each positional argument, with data variables before parameters."""

    @property
    def backend(self) -> str | None:
        """Name of the computational backend, if known."""


def _all_str(
    _: PositionalArgumentFunction, __: attrs.Attribute, value: Iterable[str]
) -> None:
    if not all(isinstance(s, str) for s in value):
        msg = f"Not all arguments are of type {str.__name__}"
        raise TypeError(msg)


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
        msg = f"There are duplicate argument names: {sorted(set(duplicate_arguments))}"
        raise ValueError(msg)


def _validate_arguments(
    instance: PositionalArgumentFunction, _: attrs.Attribute, value: Callable
) -> None:
    if not callable(value):
        msg = "Function is not callable"
        raise TypeError(msg)
    n_args = len(instance.argument_order)
    signature = inspect.signature(value)
    if len(signature.parameters) != n_args:
        if len(signature.parameters) == 1:
            parameter = next(iter(signature.parameters.values()))
            if parameter.kind == parameter.VAR_POSITIONAL:
                return
        msg = (
            f"Lambdified function expects {len(signature.parameters)} arguments, but"
            f" {n_args} sorted arguments were provided."
        )
        raise ValueError(msg)


def _to_tuple(argument_order: Iterable[str]) -> tuple[str, ...]:
    return tuple(argument_order)


@frozen
class PositionalArgumentFunction(Function[DataSample, np.ndarray]):
    """Wrapper around a function with positional arguments.

    This class provides a :meth:`~.Function.__call__` that can take a `.DataSample` for
    a function with `positional arguments
    <https://docs.python.org/3/glossary.html#term-positional-argument>`_. Its
    :attr:`argument_order` redirect the keys in the `.DataSample` to the argument
    positions in its underlying :attr:`function`.

    .. seealso:: :func:`.create_function`
    """

    function: Callable[..., np.ndarray] = field(validator=_validate_arguments)
    """A function with positional arguments only."""
    argument_order: tuple[str, ...] = field(
        converter=_to_tuple, validator=[_all_str, _all_unique]
    )
    """Ordered labels for each positional argument."""
    backend: str | None = None
    """Name of the computational backend that :attr:`function` was compiled for."""

    def __call__(self, data: DataSample) -> np.ndarray:
        args = [data[var_name] for var_name in self.argument_order]
        return self.function(*args)


class ParametrizedBackendFunction(ParametrizedFunction[DataSample, np.ndarray]):
    """Implements `.ParametrizedFunction` for a specific computational back-end.

    .. seealso:: :func:`.create_parametrized_function`
    """

    def __init__(
        self,
        function: Callable[..., np.ndarray],
        argument_order: Iterable[str],
        parameters: Mapping[str, ParameterValue],
        backend: str | None = None,
    ) -> None:
        self.__function = PositionalArgumentFunction(function, argument_order, backend)
        self.__parameters = dict(parameters)

    def __call__(
        self,
        data: DataSample,
        parameters: Mapping[str, ParameterType] | None = None,
    ) -> np.ndarray:
        extended_data: dict = {**data, **self.__parameters}
        if parameters is not None:
            self.__validate_parameters(parameters)
            extended_data.update(parameters)
        return self.__function(extended_data)

    @property
    def function(self) -> Callable[..., np.ndarray]:
        return self.__function.function

    @property
    def argument_order(self) -> tuple[str, ...]:
        return self.__function.argument_order

    @property
    def backend(self) -> str | None:
        return self.__function.backend

    @property
    def parameters(self) -> dict[str, ParameterValue]:
        return dict(self.__parameters)

    def with_parameters(
        self, parameters: Mapping[str, ParameterValue]
    ) -> ParametrizedBackendFunction:
        self.__validate_parameters(parameters)
        return ParametrizedBackendFunction(
            function=self.function,
            argument_order=self.argument_order,
            parameters={**self.__parameters, **parameters},
            backend=self.backend,
        )

    def __validate_parameters(self, parameters: Mapping[str, ParameterType]) -> None:
        over_defined = set(parameters) - set(self.__parameters)
        if over_defined:
            sep = "\n    "
            parameter_listing = f"{sep}".join(sorted(self.__parameters))
            msg = (
                f"Parameters {over_defined} do not exist in function arguments."
                f" Expecting one of:{sep}{parameter_listing}"
            )
            raise ValueError(msg)


def get_source_code(function: Function) -> str:
    """Get the backend source code used to compile this function.

    >>> import sympy as sp
    >>> from tensorwaves.function.sympy import create_function
    >>> x, y = sp.symbols("x y")
    >>> expr = x**2 + y**2
    >>> func = create_function(expr, backend="jax", use_cse=False)
    >>> src = get_source_code(func)
    >>> print(src.strip())
    def _lambdifygenerated(x, y):
        return x**2 + y**2
    """
    if subfunction := getattr(function, "function", None):
        function = subfunction
    if callable(function):
        return inspect.getsource(function)
    msg = (
        f"Cannot get source code for {Function.__name__} type {type(function).__name__}"
    )
    raise NotImplementedError(msg)
