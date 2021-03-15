"""Defines top-level interfaces of tensorwaves."""

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from expertsystem.amplitude.kinematics import ReactionInfo

# Data classes from the expertsystem do not work with jax and jit
# https://github.com/google/jax/issues/3092
# https://github.com/google/jax/issues/4416
FourMomentum = Tuple[float, float, float, float]
MomentumSample = Mapping[int, Sequence[FourMomentum]]
DataSample = Mapping[str, np.ndarray]
"""Input data for a `Function`."""


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
    def __call__(self, dataset: DataSample) -> np.ndarray:
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


class DataTransformer(ABC):
    """Interface of a data converter."""

    @abstractmethod
    def transform(self, dataset: DataSample) -> DataSample:
        """Transform a dataset into another dataset.

        This changes the keys and values of the input `.DataSample` to a
        specific output `.DataSample` structure.
        """


class Model(ABC):
    """Interface of a model which can be lambdified into a callable."""

    @abstractmethod
    def lambdify(self, backend: Union[str, tuple, dict]) -> Callable:
        """Lambdify the model into a Callable.

        Args:
          backend: Choice of backend for fast evaluations.

        The arguments of the Callable are union of the variables and parameters.
        The return value of the Callable is Any. In theory the return type
        should be a value type depending on the model. Currently, there no
        typing support is implemented for this.
        """

    @abstractmethod
    def performance_optimize(self, fix_inputs: DataSample) -> "Model":
        """Create a performance optimized model, based on fixed inputs."""

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Union[float, complex]]:
        """Get mapping of parameters to suggested initial values."""

    @property
    @abstractmethod
    def variables(self) -> FrozenSet[str]:
        """Expected input variable names."""

    @property
    def argument_order(self) -> Tuple[str, ...]:
        """Order of arguments of lambdified function signature."""


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
    ) -> np.ndarray:
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
    def setup(self, reaction_info: ReactionInfo) -> None:
        """Hook for initialization of the PhaseSpaceGenerator.

        Called before any generate calls.
        """

    @abstractmethod
    def generate(
        self, size: int, rng: UniformRealNumberGenerator
    ) -> Tuple[MomentumSample, np.ndarray]:
        """Generate phase space sample.

        Returns a `tuple` of a mapping of final state IDs to `numpy.array` s
        with four-momentum tuples.
        """
