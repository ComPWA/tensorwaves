from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from tensorwaves.interface import DataTransformer


def to_tuple(items: Iterable[DataTransformer]) -> tuple[DataTransformer, ...]:
    return tuple(items)
