# pylint: disable=consider-using-with
"""Collection of loggers that can be inserted into an optimizer as callback."""

import csv
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import IO, Any, Dict, Iterable, List, Optional, Union

import numpy as np
import tensorflow as tf
import yaml


class Loadable(ABC):
    @staticmethod
    @abstractmethod
    def load_latest_parameters(filename: str) -> dict:
        pass


class Callback(ABC):
    """Interface for callbacks such as `.CSVSummary`.

    .. seealso:: :ref:`usage/step3:Custom callbacks`
    """

    @abstractmethod
    def on_optimize_start(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    @abstractmethod
    def on_optimize_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    @abstractmethod
    def on_iteration_end(
        self, iteration: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        pass

    @abstractmethod
    def on_function_call_end(
        self, function_call: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        pass


class CallbackList(Callback):
    """Class for combining `Callback` s.

    Combine different `Callback` classes in to a chain as follows:

    >>> from tensorwaves.optimizer.callbacks import (
    ...     CallbackList, TFSummary, YAMLSummary
    ... )
    >>> from tensorwaves.optimizer.minuit import Minuit2
    >>> optimizer = Minuit2(
    ...     callback=CallbackList([TFSummary(), YAMLSummary("result.yml")])
    ... )
    """

    def __init__(self, callbacks: Iterable[Callback]) -> None:
        self.__callbacks: List[Callback] = []
        for callback in callbacks:
            self.__callbacks.append(callback)

    def on_optimize_start(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.__callbacks:
            callback.on_optimize_start(logs)

    def on_optimize_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.__callbacks:
            callback.on_optimize_end(logs)

    def on_iteration_end(
        self, iteration: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        for callback in self.__callbacks:
            callback.on_iteration_end(iteration, logs)

    def on_function_call_end(
        self, function_call: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        for callback in self.__callbacks:
            callback.on_function_call_end(function_call, logs)


class CSVSummary(Callback, Loadable):
    def __init__(
        self,
        filename: str,
        function_call_step_size: int = 1,
        iteration_step_size: Optional[int] = None,
    ) -> None:
        """Log fit parameters and the estimator value to a CSV file."""
        if iteration_step_size is None:
            iteration_step_size = 0
        if function_call_step_size <= 0 and iteration_step_size <= 0:
            raise ValueError(
                "either function call or interaction step size should > 0."
            )
        self.__function_call_step_size = function_call_step_size
        self.__iteration_step_size = iteration_step_size
        self.__latest_function_call: Optional[int] = None
        self.__latest_iteration: Optional[int] = None
        self.__writer: Optional[csv.DictWriter] = None
        self.__filename = filename
        self.__stream: IO = open(os.devnull, "w")

    def on_optimize_start(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if logs is None:
            raise ValueError(
                f"{self.__class__.__name__} requires logs on optimize start"
                " to determine header names"
            )
        if self.__function_call_step_size > 0:
            self.__latest_function_call = 0
        if self.__iteration_step_size > 0:
            self.__latest_iteration = 0
        self.__stream = open(self.__filename, "w", newline="")
        self.__writer = csv.DictWriter(
            self.__stream,
            fieldnames=list(self.__log_to_rowdict(logs)),
            quoting=csv.QUOTE_NONNUMERIC,
        )
        self.__writer.writeheader()

    def on_optimize_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if logs is not None:
            self.__latest_function_call = None
            self.__latest_iteration = None
            self.__write(logs)
        if self.__stream:
            self.__stream.close()

    def on_iteration_end(
        self, iteration: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        self.__latest_iteration = iteration
        if logs is None:
            return
        if (
            self.__iteration_step_size is None
            or self.__latest_iteration % self.__iteration_step_size != 0
        ):
            return
        self.__write(logs)

    def on_function_call_end(
        self, function_call: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        self.__latest_function_call = function_call
        if logs is None:
            return
        if (
            self.__function_call_step_size is None
            or self.__latest_function_call % self.__function_call_step_size
            != 0
        ):
            return
        self.__write(logs)

    def __write(self, logs: Dict[str, Any]) -> None:
        if self.__writer is None:
            raise ValueError(
                f"{csv.DictWriter.__name__} has not been initialized"
            )
        row_dict = self.__log_to_rowdict(logs)
        self.__writer.writerow(row_dict)

    def __log_to_rowdict(self, logs: Dict[str, Any]) -> Dict[str, Any]:
        output = {
            "time": logs["time"],
            "estimator_type": logs["estimator"]["type"],
            "estimator_value": logs["estimator"]["value"],
            **logs["parameters"],
        }
        if self.__latest_function_call is not None:
            output = {
                "function_call": self.__latest_function_call,
                **output,
            }
        if self.__latest_iteration is not None:
            output = {
                "iteration": self.__latest_iteration,
                **output,
            }
        return output

    @staticmethod
    def load_latest_parameters(filename: str) -> dict:
        def cast_non_numeric(value: str) -> Union[complex, float, str]:
            # https://docs.python.org/3/library/csv.html#csv.QUOTE_NONNUMERIC
            # does not work well for complex numbers
            try:
                return complex(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value

        with open(filename, "r") as stream:
            reader = csv.DictReader(stream)
            last_line = list(reader)[-1]
        return {
            name: cast_non_numeric(value) for name, value in last_line.items()
        }


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

        .. code-block:: shell

            tensorboard --logdir logs
        """
        self.__logdir = logdir
        self.__subdir = subdir
        self.__step_size = step_size
        self.__file_writer = open(os.devnull, "w")

    def on_optimize_start(self, logs: Optional[Dict[str, Any]] = None) -> None:
        # pylint: disable=no-member
        output_dir = (
            self.__logdir + "/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        if self.__subdir is not None:
            output_dir += "/" + self.__subdir
        self.__file_writer = tf.summary.create_file_writer(output_dir)
        self.__file_writer.set_as_default()  # type: ignore

    def on_optimize_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.__file_writer:
            self.__file_writer.close()

    def on_iteration_end(
        self, iteration: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        pass

    def on_function_call_end(
        self, function_call: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        # pylint: disable=no-member
        if logs is None:
            return
        if function_call % self.__step_size != 0:
            return
        parameters = logs["parameters"]
        for par_name, value in parameters.items():
            tf.summary.scalar(par_name, value, step=function_call)
        estimator_value = logs.get("estimator", {}).get("value", None)
        if estimator_value is not None:
            tf.summary.scalar("estimator", estimator_value, step=function_call)
        self.__file_writer.flush()


class YAMLSummary(Callback, Loadable):
    def __init__(self, filename: str, step_size: int = 10) -> None:
        """Log fit parameters and the estimator value to a `tf.summary`.

        The logs can be viewed with `TensorBoard
        <https://www.tensorflow.org/tensorboard>`_ via:

        .. code-block:: shell

            tensorboard --logdir logs
        """
        self.__step_size = step_size
        self.__filename = filename
        self.__stream: IO = open(os.devnull, "w")

    def on_optimize_start(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.__stream = open(self.__filename, "w")

    def on_optimize_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if logs is None:
            return
        self.__dump_to_yaml(logs)
        self.__stream.close()

    def on_iteration_end(
        self, iteration: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        pass

    def on_function_call_end(
        self, function_call: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        if logs is None:
            return
        if function_call % self.__step_size != 0:
            return
        self.__dump_to_yaml(logs)

    def __dump_to_yaml(self, logs: Dict[str, Any]) -> None:
        _empty_file(self.__stream)
        cast_logs = dict(logs)
        cast_logs["parameters"] = {
            p: _cast_value(v) for p, v in logs["parameters"].items()
        }
        yaml.dump(
            cast_logs,
            self.__stream,
            sort_keys=False,
            Dumper=_IncreasedIndent,
            default_flow_style=False,
        )

    @staticmethod
    def load_latest_parameters(filename: str) -> dict:
        with open(filename) as stream:
            fit_stats = yaml.load(stream, Loader=yaml.Loader)
        return fit_stats["parameters"]


def _cast_value(value: Any) -> Union[complex, float]:
    # cspell:ignore iscomplex
    if np.iscomplex(value) or isinstance(value, complex):
        return complex(value)
    return float(value)


class _IncreasedIndent(yaml.Dumper):
    # pylint: disable=too-many-ancestors
    def increase_indent(self, flow=False, indentless=False):  # type: ignore
        return super().increase_indent(flow, False)


def _empty_file(stream: IO) -> None:
    stream.seek(0)
    stream.truncate()
