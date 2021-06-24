"""Defines top-level interface of tensorwaves."""

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

import attr
import numpy as np
from attr.validators import instance_of, optional

try:
    from IPython.lib.pretty import PrettyPrinter  # type: ignore
except ImportError:
    PrettyPrinter = Any

# Custom classes do not work with jax and jit
# https://github.com/google/jax/issues/3092
# https://github.com/google/jax/issues/4416
FourMomentum = Tuple[float, float, float, float]
r"""Tuple of energy and three-momentum :math:`(E, \vec{p})`."""
MomentumSample = Mapping[int, Sequence[FourMomentum]]
"""Mapping of state ID to a event-wise sequence of their `.FourMomentum`."""
DataSample = Mapping[str, np.ndarray]
"""Mapping of variable names to a sequence of data points, used by `Function`."""

ParameterValue = Union[complex, float]
"""Allowed types for parameter values."""


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
            ReactionInfo of the function evaluation. Type depends on the input type.
        """

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, ParameterValue]:
        """Get `dict` of parameters."""

    @abstractmethod
    def update_parameters(
        self, new_parameters: Mapping[str, ParameterValue]
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
    def parameters(self) -> Dict[str, ParameterValue]:
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
    def __call__(self, parameters: Mapping[str, ParameterValue]) -> float:
        """Evaluate discrepancy."""

    @abstractmethod
    def gradient(
        self, parameters: Mapping[str, ParameterValue]
    ) -> Dict[str, ParameterValue]:
        """Calculate gradient for given parameter mapping."""


_PARAMETER_DICT_VALIDATOR = attr.validators.deep_mapping(
    key_validator=instance_of(str),
    mapping_validator=instance_of(dict),
    value_validator=instance_of(ParameterValue.__args__),  # type: ignore
)


@attr.s(frozen=True)
class FitResult:  # pylint: disable=too-many-instance-attributes
    minimum_valid: bool = attr.ib(validator=instance_of(bool))
    execution_time: float = attr.ib(validator=instance_of(float))
    function_calls: int = attr.ib(validator=instance_of(int))
    estimator_value: float = attr.ib(validator=instance_of(float))
    parameter_values: Dict[str, ParameterValue] = attr.ib(
        default=None, validator=_PARAMETER_DICT_VALIDATOR
    )
    parameter_errors: Optional[Dict[str, ParameterValue]] = attr.ib(
        default=None, validator=optional(_PARAMETER_DICT_VALIDATOR)
    )
    iterations: Optional[int] = attr.ib(
        default=None, validator=optional(instance_of(int))
    )
    specifics: Optional[Any] = attr.ib(default=None)
    """Any additional info provided by the specific optimizer."""

    @parameter_errors.validator  # pyright: reportOptionalMemberAccess=false
    def _check_parameter_errors(
        self, _: attr.Attribute, value: Optional[Dict[str, ParameterValue]]
    ) -> None:
        if value is None:
            return
        for par_name in value:
            if par_name not in self.parameter_values:
                raise ValueError(
                    f'No parameter value exists for parameter error "{par_name}"'
                )

    def _repr_pretty_(self, p: PrettyPrinter, cycle: bool) -> None:
        class_name = type(self).__name__
        if cycle:
            p.text(f"{class_name}(...)")
        else:
            with p.group(indent=1, open=f"{class_name}("):
                for field in attr.fields(type(self)):
                    if field.name in {"specifics"}:
                        continue
                    value = getattr(self, field.name)
                    if value != field.default:
                        p.breakable()
                        p.text(f"{field.name}=")
                        if isinstance(value, dict):
                            with p.group(indent=1, open="{"):
                                for key, val in value.items():
                                    p.breakable()
                                    p.pretty(key)
                                    p.text(": ")
                                    p.pretty(val)
                                    p.text(",")
                            p.breakable()
                            p.text("}")
                        else:
                            p.pretty(value)
                        p.text(",")
            p.breakable()
            p.text(")")

    def count_number_of_parameters(self, complex_twice: bool = False) -> int:
        """Compute the number of free parameters in a `.FitResult`.

        Args:
            fit_result (FitResult): Fit result from which to count it's
                `~.FitResult.parameter_values`.


            complex_twice (bool): Count complex-valued parameters twice.
        """
        n_parameters = len(self.parameter_values)
        if complex_twice:
            complex_values = filter(
                lambda v: isinstance(v, complex),
                self.parameter_values.values(),
            )
            n_parameters += len(list(complex_values))
        return n_parameters


class Optimizer(ABC):
    """Optimize a fit model to a data set."""

    @abstractmethod
    def optimize(
        self,
        estimator: Estimator,
        initial_parameters: Mapping[str, ParameterValue],
    ) -> FitResult:
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
    def setup(
        self,
        initial_state_mass: float,
        final_state_masses: Mapping[int, float],
    ) -> None:
        """Hook for initialization of the `.PhaseSpaceGenerator`.

        Called before any :meth:`.generate` calls.

        Args:
            initial_state_mass: Mass of the decaying state.
            final_state_masses: A mapping of final state IDs to the
                corresponding masses.
        """

    @abstractmethod
    def generate(
        self, size: int, rng: UniformRealNumberGenerator
    ) -> Tuple[MomentumSample, np.ndarray]:
        """Generate phase space sample.

        Returns:
            A `tuple` of a `.MomentumSample` plus a event-wise sequence of
            weights.
        """
