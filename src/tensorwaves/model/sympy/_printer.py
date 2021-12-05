# pylint: disable=abstract-method, invalid-name, protected-access
import re
from typing import Dict

from sympy.printing.numpy import NumPyPrinter


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


class TensorflowPrinter(CustomNumPyPrinter):
    module_imports = {"tensorflow.experimental": {"numpy as tnp"}}
    _module = "tnp"

    def __init__(self) -> None:
        # https://github.com/sympy/sympy/blob/f1384c2/sympy/printing/printer.py#L21-L72
        super().__init__()
        self.known_functions["ComplexSqrt"] = "sqrt"
        self.printmethod = "_tensorflow_code"
