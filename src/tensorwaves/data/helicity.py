"""Definition of a DataConverter based on a HelicityAdapter."""

from expertsystem.amplitude.kinematics import EventCollection, HelicityAdapter

from tensorwaves.interfaces import DataConverter, DataSample


class HelicityKinematicsConverter(DataConverter):
    def __init__(self, helicity_adapter: HelicityAdapter) -> None:
        self.__helicity_adapter = helicity_adapter

    def convert(self, dataset: DataSample) -> DataSample:
        events = EventCollection({int(k): v for k, v in dataset.items()})
        return self.__helicity_adapter.convert(events)
