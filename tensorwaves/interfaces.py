from abc import ABC, abstractmethod


class Function(ABC):
    """
    Defines the interface of a callable function.
    The parameters of the model are separated from the domain variables. This
    follows the mathematical definition, in which a function defines its domain
    and parameters. However specific points in the domain are not relevant.
    Hence while the domain variables are the argument of the evaluation
    (:func:`~Function.__call__`), the parameters are controlled via a getter
    and setter (see :func:`~Function.parameters`). The reason for this
    separation is to facilitate the events when parameters have changed.
    """
    @abstractmethod
    def __call__(self, dataset: dict):
        """
        """

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """
        """

    @parameters.setter
    @abstractmethod
    def parameters(self, new_parameters: dict):
        """
        """


class Estimator(ABC):
    @abstractmethod
    def __call__(self) -> float:
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict:
        pass

    @parameters.setter
    @abstractmethod
    def parameters(self, new_parameters: dict):
        pass


class Kinematics(ABC):
    @abstractmethod
    def convert(self, events: dict) -> dict:
        pass

    @abstractmethod
    def reduce_to_phase_space(self, events: dict) -> dict:
        pass

    @property
    @abstractmethod
    def phase_space_volume(self) -> float:
        pass


class Optimizer(ABC):
    @abstractmethod
    def optimize(self, estimator: Estimator, initial_parameters: dict) -> dict:
        pass


class PhaseSpaceGenerator(ABC):
    @abstractmethod
    def generate(self, size: int, random_generator) -> dict:
        pass


class UniformRealNumberGenerator(ABC):
    @abstractmethod
    def __call__(self, size: int) -> float or list:
        '''
        generate random floats in the range from [0,1)
        '''

    @property
    @abstractmethod
    def seed(self) -> dict:
        pass

    @seed.setter
    @abstractmethod
    def seed(self, value: float):
        pass
