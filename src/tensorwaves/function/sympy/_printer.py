# pylint: disable=abstract-method protected-access
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Callable, Iterable, TypeVar

from sympy.printing.numpy import NumPyPrinter  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    import sympy as sp


def _replace_module(
    mapping: dict[str, str], old: str, new: str
) -> dict[str, str]:
    return {
        k: re.sub(rf"^{old}\.(.*)$", rf"{new}.\1", v)
        for k, v in mapping.items()
    }


class CustomNumPyPrinter(NumPyPrinter):
    def __init__(self) -> None:
        # https://github.com/sympy/sympy/blob/f291f2d/sympy/utilities/lambdify.py#L821-L823
        super().__init__(
            settings={
                "fully_qualified_modules": False,
                "inline": True,
                "allow_unknown_functions": True,
            }
        )
        self._kc = _replace_module(NumPyPrinter._kc, "numpy", self._module)
        self._kf = _replace_module(NumPyPrinter._kf, "numpy", self._module)
        self.printmethod = "_numpycode"  # force using _numpycode methods


class JaxPrinter(CustomNumPyPrinter):
    module_imports = {"jax": {"numpy as jnp"}}
    _module = "jnp"


_T = TypeVar("_T")


def _forward_to_numpy_printer(
    class_names: Iterable[str],
) -> Callable[[type[_T]], type[_T]]:
    """Decorator for a `~sympy.printing.printer.Printer` class.

    Args:
        class_names: The names of classes that should be printed with their
            :code:`_numpycode()` method.
    """

    def decorator(decorated_class: type[_T]) -> type[_T]:
        def _get_numpy_code(self: _T, expr: sp.Expr, *args: Any) -> str:
            return expr._numpycode(self, *args)  # type: ignore[attr-defined]

        for class_name in class_names:
            method_name = f"_print_{class_name}"
            setattr(decorated_class, method_name, _get_numpy_code)
        return decorated_class

    return decorator


@_forward_to_numpy_printer(
    [
        "ArrayAxisSum",
        "ArrayMultiplication",
        "BoostZ",
        "BoostZMatrix",
        "RotationY",
        "RotationYMatrix",
        "RotationZ",
        "RotationZMatrix",
        "_ArraySize",
        "_BoostZMatrixImplementation",
        "_OnesArray",
        "_RotationYMatrixImplementation",
        "_RotationZMatrixImplementation",
        "_ZerosArray",
    ]
)
class TensorflowPrinter(CustomNumPyPrinter):
    module_imports = {"tensorflow.experimental": {"numpy as tnp"}}
    _module = "tnp"

    def __init__(self) -> None:
        # https://github.com/sympy/sympy/blob/f1384c2/sympy/printing/printer.py#L21-L72
        super().__init__()
        self.known_functions["ComplexSqrt"] = "sqrt"
        self.printmethod = "_tensorflow_code"
