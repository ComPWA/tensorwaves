"""Collection of loggers that can be inserted into an optimizer as callback."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import IO, Iterable, List, Optional

import pandas as pd
import tensorflow as tf
import yaml

from tensorwaves.interfaces import Estimator


class Callback(ABC):
    @abstractmethod
    def on_iteration_end(
        self, parameters: dict, estimator_value: float
    ) -> None:
        pass

    @abstractmethod
    def on_function_call_end(self) -> None:
        pass


class CallbackList(Callback):
    """Class for combining `Callback` s.

    Combine different `Callback` classes in to a chain as follows:

    >>> from tensorwaves.optimizer.callbacks import (
    ...     CallbackList, TFSummary, YAMLSummary
    ... )
    >>> from tensorwaves.optimizer.minuit import Minuit2
    >>> optimizer = Minuit2(callback=CallbackList([TFSummary(), YAMLSummary()]))
    """

    def __init__(self, callbacks: Iterable[Callback]) -> None:
        self.__callbacks: List[Callback] = list()
        for callback in callbacks:
            self.__callbacks.append(callback)

    def on_iteration_end(
        self, parameters: dict, estimator_value: float
    ) -> None:
        for callback in self.__callbacks:
            callback.on_iteration_end(parameters, estimator_value)

    def on_function_call_end(self) -> None:
        for callback in self.__callbacks:
            callback.on_function_call_end()


class CSVSummary(Callback):
    def __init__(
        self,
        filename: str,
        estimator: Estimator,
        step_size: int = 10,
    ) -> None:
        """Log fit parameters and the estimator value to a CSV file."""
        self.__function_call = -1
        self.__step_size = step_size
        self.__first_call = True
        self.__stream = open(filename, "w")
        _empty_file(self.__stream)
        if not isinstance(estimator, Estimator):
            raise TypeError(f"Requires an in {Estimator.__name__} instance")
        self.__estimator_type: str = estimator.__class__.__name__

    def on_iteration_end(
        self, parameters: dict, estimator_value: float
    ) -> None:
        self.__function_call += 1
        if self.__function_call % self.__step_size != 0:
            return
        output_dict = {
            "time": datetime.now(),
            "function_call": self.__function_call,
            "estimator_type": self.__estimator_type,
            "estimator_value": float(estimator_value),
        }
        output_dict.update(
            {name: float(value) for name, value in parameters.items()}
        )

        data_frame = pd.DataFrame(output_dict, index=[self.__function_call])
        data_frame.to_csv(
            self.__stream,
            mode="a",
            header=self.__first_call,
            index=False,
        )
        self.__first_call = False

    def on_function_call_end(self) -> None:
        self.__stream.close()


class TFSummary(Callback):
    def __init__(
        self,
        logdir: str = "logs",
        step_size: int = 10,
        subdir: Optional[str] = None,
    ) -> None:
        """Log fit parameters and the estimator value to a `tf.summary`.

        The logs can be viewed with `TensorBoard
        <https://www.tensorflow.org/tensorboard>`_ via:

        .. code-block:: bash

            tensorboard --logdir logs
        """
        output_dir = logdir + "/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        if subdir is not None:
            output_dir += "/" + subdir
        self.__file_writer = tf.summary.create_file_writer(output_dir)
        self.__file_writer.set_as_default()
        self.__function_call = 0
        self.__step_size = step_size

    def on_iteration_end(
        self, parameters: dict, estimator_value: float
    ) -> None:
        self.__function_call += 1
        if self.__function_call % self.__step_size != 0:
            return
        for par_name, value in parameters.items():
            tf.summary.scalar(par_name, value, step=self.__function_call)
        tf.summary.scalar(
            "estimator", estimator_value, step=self.__function_call
        )
        self.__file_writer.flush()

    def on_function_call_end(self) -> None:
        self.__file_writer.close()


class YAMLSummary(Callback):
    def __init__(
        self,
        filename: str,
        estimator: Estimator,
        step_size: int = 10,
    ) -> None:
        """Log fit parameters and the estimator value to a `tf.summary`.

        The logs can be viewed with `TensorBoard
        <https://www.tensorflow.org/tensorboard>`_ via:

        .. code-block:: bash

            tensorboard --logdir logs
        """
        self.__function_call = 0
        self.__step_size = step_size
        self.__stream = open(filename, "w")
        if not isinstance(estimator, Estimator):
            raise TypeError(f"Requires an in {Estimator.__name__} instance")
        self.__estimator_type: str = estimator.__class__.__name__

    def on_iteration_end(
        self, parameters: dict, estimator_value: float
    ) -> None:
        self.__function_call += 1
        if self.__function_call % self.__step_size != 0:
            return
        output_dict = {
            "Time": datetime.now(),
            "FunctionCalls": self.__function_call,
            "Estimator": {
                "Type": self.__estimator_type,
                "Value": float(estimator_value),
            },
            "Parameters": {
                name: float(value) for name, value in parameters.items()
            },
        }
        _empty_file(self.__stream)
        yaml.dump(
            output_dict,
            self.__stream,
            sort_keys=False,
            Dumper=_IncreasedIndent,
            default_flow_style=False,
        )

    def on_function_call_end(self) -> None:
        self.__stream.close()


class _IncreasedIndent(yaml.Dumper):
    # pylint: disable=too-many-ancestors
    def increase_indent(self, flow=False, indentless=False):  # type: ignore
        return super().increase_indent(flow, False)


def _empty_file(stream: IO) -> None:
    stream.seek(0)
    stream.truncate()
