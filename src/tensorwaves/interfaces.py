"""Defines top-level interfaces of tensorwaves."""

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional, Tuple, Union


class Function(ABC):
    """Interface of a callable function.

    The parameters of the model are separated from the domain variables. This
    follows the mathematical definition, in which a function defines its domain
    and parameters. However specific points in the domain are not relevant.
    Hence while the domain variables are the argument of the evaluation
    (see :func:`~Function.__call__`), the parameters are controlled via a
    getter and setter (see :func:`~Function.parameters`). The reason for this
    separation is to facilitate the events when parameters have changed.
    """

    @abstractmethod
    def __call__(self, dataset: dict) -> list:
        """Evaluate the function.

        Args:
            dataset: a `dict` with domain variable names as keys.

        Return:
            `list` or `numpy.array` of values.

        """

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """Get `dict` of parameters."""

    @abstractmethod
    def update_parameters(self, new_parameters: dict) -> None:
        """Update the collection of parameters."""


class Estimator(ABC):
    """Estimator for discrepancy model and data."""

    @abstractmethod
    def __call__(self, parameters: Dict[str, float]) -> float:
        """Evaluate discrepancy."""

    @property
    @abstractmethod
    def parameters(self) -> Iterable[str]:
        """Get list of parameter names."""


class Kinematics(ABC):
    """Abstract interface for computation of kinematic variables."""

    @abstractmethod
    def convert(self, events: dict) -> dict:
        """Convert a set of momentum tuples (events) to kinematic variables."""

    @abstractmethod
    def is_within_phase_space(self, events: dict) -> Tuple[bool]:
        """Check which events lie within phase space."""

    @property
    @abstractmethod
    def phase_space_volume(self) -> float:
        """Compute volume of the phase space."""


class Optimizer(ABC):
    """Optimize a fit model to a data set."""

    @abstractmethod
    def optimize(self, estimator: Estimator, initial_parameters: dict) -> dict:
        """Execute optimization."""


class UniformRealNumberGenerator(ABC):
    """Abstract class for generating uniform real numbers."""

    @abstractmethod
    def __call__(
        self, size: int, min_value: float = 0.0, max_value: float = 1.0
    ) -> Union[float, list]:
        """Generate random floats in the range from [min_value,max_value)."""

    @property  # type: ignore
    @abstractmethod
    def seed(self) -> Optional[float]:
        """Get random seed. `None` if you want indeterministic behavior."""

    @seed.setter  # type: ignore
    @abstractmethod
    def seed(self, value: Optional[float]) -> None:
        """Set random seed. Use `None` for indeterministic behavior."""


class PhaseSpaceGenerator(ABC):
    """Abstract class for generating phase space samples."""

    @abstractmethod
    def generate(self, size: int, rng: UniformRealNumberGenerator) -> dict:
        """Generate phase space sample."""
