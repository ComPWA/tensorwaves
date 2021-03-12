"""Implementations of `.DataConverter`."""

from expertsystem.amplitude.kinematics import EventCollection, HelicityAdapter

from tensorwaves.interfaces import DataConverter, DataSample


class HelicityKinematicsConverter(DataConverter):
    """Convert four-momentum tuples to variables in the helicity formalism.

    Implementation of a `.DataConverter` based on the
    `~expertsystem.amplitude.kinematics.HelicityAdapter`.
    """

    def __init__(self, helicity_adapter: HelicityAdapter) -> None:
        self.__helicity_adapter = helicity_adapter

    def convert(self, dataset: DataSample) -> DataSample:
        events = EventCollection({int(k): v for k, v in dataset.items()})
        return self.__helicity_adapter.convert(events)
