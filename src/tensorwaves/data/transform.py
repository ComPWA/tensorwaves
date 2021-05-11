"""Implementations of `.DataTransformer`."""

import numpy as np
from ampform.kinematics import EventCollection, HelicityAdapter

from tensorwaves.interfaces import DataSample, DataTransformer


class HelicityTransformer(DataTransformer):
    """Transform four-momentum tuples to variables in the helicity formalism.

    Implementation of a `.DataTransformer` based on the
    `~ampform.kinematics.HelicityAdapter`.
    """

    def __init__(self, helicity_adapter: HelicityAdapter) -> None:
        self.__helicity_adapter = helicity_adapter

    def transform(self, dataset: DataSample) -> DataSample:
        events = EventCollection({int(k): v for k, v in dataset.items()})
        dataset = self.__helicity_adapter.transform(events)
        return {key: np.array(values) for key, values in dataset.items()}
