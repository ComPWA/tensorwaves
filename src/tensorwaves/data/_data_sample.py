"""Helper functions for modifying `.DataSample` instances."""

from typing import Any

import numpy as np
from tqdm.auto import tqdm

from tensorwaves.interface import DataSample


def get_number_of_events(four_momenta: DataSample) -> int:
    if len(four_momenta) == 0:
        return 0
    return len(next(iter(four_momenta.values())))


def concatenate_events(sample1: DataSample, sample2: DataSample) -> DataSample:
    if len(sample1) and len(sample2) and set(sample1) != set(sample2):
        raise ValueError(
            "Keys of data sets are not matching", set(sample2), set(sample1)
        )
    if get_number_of_events(sample1) == 0:
        return sample2
    return {
        i: np.concatenate((array, sample2[i])) for i, array in sample1.items()
    }


def stack_events(sample1: DataSample, sample2: DataSample) -> DataSample:
    if len(sample1) and len(sample2) and set(sample1) != set(sample2):
        raise ValueError(
            "Keys of data sets are not matching", set(sample2), set(sample1)
        )
    if get_number_of_events(sample1) == 0:
        return sample2
    return {i: np.vstack((array, sample2[i])) for i, array in sample1.items()}


def select_events(four_momenta: DataSample, selector: Any) -> DataSample:
    return {i: values[selector] for i, values in four_momenta.items()}


def finalize_progress_bar(progress_bar: tqdm) -> None:
    remainder = progress_bar.total - progress_bar.n
    progress_bar.update(n=remainder)  # pylint crashes if total is set directly
    progress_bar.close()
