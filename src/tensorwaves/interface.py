"""Defines top-level interface of tensorwaves."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Mapping, Optional, Tuple, TypeVar, Union

import attr
import numpy as np
from attr.validators import instance_of, optional

try:
    # pyright: reportMissingImports=false
    from IPython.lib.pretty import PrettyPrinter
except ImportError:
    PrettyPrinter = Any


InputType = TypeVar("InputType")
"""The argument type of a :meth:`.Function.__call__`."""
OutputType = TypeVar("OutputType")
"""The return type of a :meth:`.Function.__call__`."""


class Function(ABC, Generic[InputType, OutputType]):
    @abstractmethod
    def __call__(self, data: InputType) -> OutputType:
        ...


DataSample = Mapping[Union[int, str], np.ndarray]
"""Mapping of variable names to a sequence of data points, used by `Function`."""
ParameterValue = Union[complex, float]
"""Allowed types for parameter values."""


class ParametrizedFunction(Function[DataSample, np.ndarray]):
    """Interface of a callable function.

    The parameters of the model are separated from the domain variables. This
    follows the mathematical definition, in which a function defines its domain
    and parameters. However specific points in the domain are not relevant.
    Hence while the domain variables are the argument of the evaluation (see
    :func:`~Function.__call__`), the parameters are controlled via a getter and
    setter (see :func:`~ParametrizedFunction.parameters`). The reason for this
    separation is to facilitate the events when parameters have changed.
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


class DataTransformer(Function[DataSample, DataSample]):
    """Transform one `.DataSample` into another `.DataSample`.

    This changes the keys and values of the input `.DataSample` to a
    specific output `.DataSample` structure.
    """


class Estimator(Function[Mapping[str, ParameterValue], float]):
    """Estimator for discrepancy model and data."""

    @abstractmethod
    def gradient(
        self, parameters: Mapping[str, ParameterValue]
    ) -> Dict[str, ParameterValue]:
        """Calculate gradient for given parameter mapping."""


_PARAMETER_DICT_VALIDATOR = attr.validators.deep_mapping(
    key_validator=instance_of(str),
    mapping_validator=instance_of(dict),
    value_validator=instance_of(ParameterValue.__args__),  # type: ignore[attr-defined]
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
    """Any additional info provided by the specific optimizer.

    An instance returned by one of the implemented optimizers under the
    :mod:`.optimizer` module. Currently one of:

    - `iminuit.Minuit`
    - `scipy.optimize.OptimizeResult`

    This way, you can for instance get the `~iminuit.Minuit.covariance` matrix.
    See also :ref:`usage/step3:Covariance matrix`.
    """

    @parameter_errors.validator  # pyright: reportOptionalMemberAccess=false
    def _check_parameter_errors(
        self, _: attr.Attribute, value: Optional[Dict[str, ParameterValue]]
    ) -> None:
        if value is None:
            return
        for par_name in value:
            if par_name not in self.parameter_values:
                raise ValueError(
                    "No parameter value exists for parameter error"
                    f' "{par_name}"'
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

    @property  # type: ignore[misc]
    @abstractmethod
    def seed(self) -> Optional[float]:
        """Get random seed. `None` if you want indeterministic behavior."""

    @seed.setter  # type: ignore[misc]
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
    ) -> Tuple[DataSample, np.ndarray]:
        r"""Generate phase space sample.

        Returns:
            A `tuple` of a `.DataSample` (**four-momenta**) with an event-wise
            sequence of weights. The four-momenta are arrays of shape
            :math:`n \times 4`.
        """
