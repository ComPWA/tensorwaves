"""Collection of loggers that can be inserted into an optimizer as callback."""

import csv
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import IO, Any, Dict, Iterable, List, Optional

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
        self.__callbacks: List[Callback] = list()
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
    def __init__(self, filename: str, step_size: int = 1) -> None:
        """Log fit parameters and the estimator value to a CSV file."""
        self.__step_size = step_size
        self.__writer: Optional[csv.DictWriter] = None
        self.__filename = filename
        self.__stream: IO = open(os.devnull, "w")

    def on_optimize_start(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if logs is None:
            raise ValueError(
                f"{self.__class__.__name__} requires logs on optimize start"
                " to determine header names"
            )
        self.__stream = open(self.__filename, "w", newline="")
        fieldnames = list(self.__log_to_rowdict(function_call=0, logs=logs))
        self.__writer = csv.DictWriter(
            self.__stream, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC
        )
        self.__writer.writeheader()

    def on_optimize_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if logs is not None:
            self.__write(function_call=None, logs=logs)
        if self.__stream:
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
        self.__write(function_call, logs)

    def __write(
        self, function_call: Optional[int], logs: Dict[str, Any]
    ) -> None:
        if self.__writer is None:
            raise ValueError(
                f"{csv.DictWriter.__name__} has not been initialized"
            )
        row_dict = self.__log_to_rowdict(function_call, logs)
        self.__writer.writerow(row_dict)

    @staticmethod
    def __log_to_rowdict(
        function_call: Optional[int], logs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "function_call": function_call,
            "time": logs["time"],
            "estimator_type": logs["estimator"]["type"],
            "estimator_value": logs["estimator"]["value"],
            **logs["parameters"],
        }

    @staticmethod
    def load_latest_parameters(filename: str) -> dict:
        with open(filename, "r") as stream:
            reader = csv.DictReader(stream, quoting=csv.QUOTE_NONNUMERIC)
            last_line = list(reader)[-1]
        return last_line


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
        self.__stream.close()

    def on_iteration_end(
        self, iteration: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        pass

    def on_function_call_end(
        self, function_call: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        if function_call % self.__step_size != 0:
            return
        _empty_file(self.__stream)
        yaml.dump(
            logs,
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


class _IncreasedIndent(yaml.Dumper):
    # pylint: disable=too-many-ancestors
    def increase_indent(self, flow=False, indentless=False):  # type: ignore
        return super().increase_indent(flow, False)


def _empty_file(stream: IO) -> None:
    stream.seek(0)
    stream.truncate()
