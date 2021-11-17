"""Implementations of `.DataTransformer`."""

from typing import TYPE_CHECKING

import numpy as np

from tensorwaves.interface import DataSample, DataTransformer

if TYPE_CHECKING:
    from ampform.kinematics import HelicityAdapter


class HelicityTransformer(DataTransformer):
    """Transform four-momentum tuples to variables in the helicity formalism.

    Implementation of a `.DataTransformer` based on the
    `~ampform.kinematics.HelicityAdapter`.
    """

    def __init__(self, helicity_adapter: "HelicityAdapter") -> None:
        self.__helicity_adapter = helicity_adapter

    def __call__(self, dataset: DataSample) -> DataSample:
        # pylint: disable=import-outside-toplevel
        from ampform.kinematics import EventCollection

        events = EventCollection({int(k[1:]): v for k, v in dataset.items()})
        transformed_data = self.__helicity_adapter.transform(events)
        return {
            key: np.array(values) for key, values in transformed_data.items()
        }
