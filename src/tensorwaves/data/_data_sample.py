"""Helper functions for modifying `.DataSample` instances."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from tqdm.auto import tqdm

    from tensorwaves.interface import DataSample


def get_number_of_events(four_momenta: DataSample) -> int:
    if len(four_momenta) == 0:
        return 0
    return len(next(iter(four_momenta.values())))


def merge_events(sample1: DataSample, sample2: DataSample) -> DataSample:
    merge_method = _determine_merge_method(sample1)
    return _merge_events(sample1, sample2, merge_method)


def _determine_merge_method(
    sample: DataSample,
) -> Callable[[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    if len(sample) == 0:
        return operator.itemgetter(1)
    some_array = next(iter(sample.values()))
    rank = len(some_array.shape)
    if rank == 1:
        return np.concatenate
    if rank > 1:
        return np.vstack
    msg = f"Cannot find a merge method for data samples of rank {rank}"
    raise NotImplementedError(msg)


def _merge_events(
    sample1: DataSample,
    sample2: DataSample,
    merge_method: Callable[[tuple[np.ndarray, np.ndarray]], np.ndarray],
) -> DataSample:
    if len(sample1) and len(sample2) and set(sample1) != set(sample2):
        msg = "Keys of data sets are not matching"
        raise ValueError(msg, set(sample2), set(sample1))
    if get_number_of_events(sample1) == 0:
        return sample2
    return {i: merge_method((array, sample2[i])) for i, array in sample1.items()}


def select_events(four_momenta: DataSample, selector: Any) -> DataSample:
    return {i: values[selector] for i, values in four_momenta.items()}


def finalize_progress_bar(progress_bar: tqdm) -> None:
    if progress_bar.total is not None:
        remainder = progress_bar.total - progress_bar.n
    else:
        remainder = 0
    progress_bar.update(n=remainder)
    progress_bar.close()
