# pylint: disable=abstract-method protected-access
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Type, TypeVar

from sympy.printing.numpy import NumPyPrinter  # noqa: E402

if TYPE_CHECKING:
    import sympy as sp


def _replace_module(
    mapping: Dict[str, str], old: str, new: str
) -> Dict[str, str]:
    return {
        k: re.sub(fr"^{old}\.(.*)$", fr"{new}.\1", v)
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
) -> Callable[[Type[_T]], Type[_T]]:
    """Decorator for a `~sympy.printing.printer.Printer` class.

    Args:
        class_names: The names of classes that should be printed with their
            :code:`_numpycode()` method.
    """

    def decorator(decorated_class: Type[_T]) -> Type[_T]:
        def _get_numpy_code(self: _T, expr: "sp.Expr", *args: Any) -> str:
            return expr._numpycode(self, *args)

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
        "RotationY",
        "RotationZ",
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
