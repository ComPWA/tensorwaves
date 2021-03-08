"""Defines top-level interfaces of tensorwaves."""

from abc import ABC, abstractmethod
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple, Union

from expertsystem.amplitude.data import DataSet, MomentumPool, ScalarSequence


class Function(ABC):
    """Interface of a callable function.

    The parameters of the model are separated from the domain variables. This
    follows the mathematical definition, in which a function defines its domain
    and parameters. However specific points in the domain are not relevant.
    Hence while the domain variables are the argument of the evaluation (see
    :func:`~Function.__call__`), the parameters are controlled via a getter and
    setter (see :func:`~Function.parameters`). The reason for this separation
    is to facilitate the events when parameters have changed.
    """

    @abstractmethod
    def __call__(self, dataset: DataSet) -> ScalarSequence:
        """Evaluate the function.

        Args:
            dataset: a `dict` with domain variable names as keys.

        Return:
            Result of the function evaluation. Type depends on the input type.
        """

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Union[float, complex]]:
        """Get `dict` of parameters."""

    @abstractmethod
    def update_parameters(
        self, new_parameters: Mapping[str, Union[float, complex]]
    ) -> None:
        """Update the collection of parameters."""


class Model(ABC):
    """Interface of a model which can be lambdified into a callable."""

    @abstractmethod
    def lambdify(self, backend: Union[str, tuple, dict]) -> Function:
        """Lambdify the model into a Function.

        Args:
          backend: Choice of backend for fast evaluations. Argument is passed to
            the `~.lambdify` function.

        The argument of the Function resembles a dataset in the form of a
        mapping of a variable name to value type. The return value of the
        Function is of a value type.
        """

    @abstractmethod
    def performance_optimize(self, fix_inputs: DataSet) -> "Model":
        """Create a performance optimized model, based on fixed inputs."""

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Union[float, complex]]:
        """Get mapping of parameters to suggested initial values."""

    @property
    @abstractmethod
    def variables(self) -> FrozenSet[str]:
        """Expected input variable names."""


class Estimator(ABC):
    """Estimator for discrepancy model and data."""

    @abstractmethod
    def __call__(
        self, parameters: Mapping[str, Union[float, complex]]
    ) -> float:
        """Evaluate discrepancy."""

    @abstractmethod
    def gradient(
        self, parameters: Mapping[str, Union[float, complex]]
    ) -> Dict[str, Union[float, complex]]:
        """Calculate gradient for given parameter mapping."""


class Kinematics(ABC):
    """Abstract interface for computation of kinematic variables."""

    @abstractmethod
    def convert(self, events: MomentumPool) -> DataSet:
        """Convert a set of momentum tuples (events) to kinematic variables."""

    @abstractmethod
    def is_within_phase_space(self, events: MomentumPool) -> Tuple[bool]:
        """Check which events lie within phase space."""

    @property
    @abstractmethod
    def phase_space_volume(self) -> float:
        """Compute volume of the phase space."""


class Optimizer(ABC):
    """Optimize a fit model to a data set."""

    @abstractmethod
    def optimize(
        self,
        estimator: Estimator,
        initial_parameters: Mapping[str, Union[float, complex]],
    ) -> Dict[str, Any]:
        """Execute optimization."""


class UniformRealNumberGenerator(ABC):
    """Abstract class for generating uniform real numbers."""

    @abstractmethod
    def __call__(
        self, size: int, min_value: float = 0.0, max_value: float = 1.0
    ) -> ScalarSequence:
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
    def generate(
        self, size: int, rng: UniformRealNumberGenerator
    ) -> Tuple[MomentumPool, ScalarSequence]:
        """Generate phase space sample.

        Returns a `tuple` of a mapping of final state IDs to `numpy.array` s
        with four-momentum tuples.
        """
