"""Collection of shared functionality to handle an amplitude recipe file."""

from typing import (
    Any,
    Dict,
    Union,
)


def extract_value(definition: Union[float, Dict[str, Any]]) -> float:
    if isinstance(definition, (float, int)):
        return definition
    return float(definition["Value"])
