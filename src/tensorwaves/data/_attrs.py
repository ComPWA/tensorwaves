from __future__ import annotations

from typing import Iterable

from tensorwaves.interface import DataTransformer


def to_tuple(items: Iterable[DataTransformer]) -> tuple[DataTransformer, ...]:
    return tuple(items)
