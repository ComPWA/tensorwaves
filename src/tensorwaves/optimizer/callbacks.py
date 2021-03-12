"""Collection of loggers that can be inserted into an optimizer as callback."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import IO, Any, Dict, Iterable, List, Optional

import pandas as pd
import tensorflow as tf
import yaml


class Loadable(ABC):
    @staticmethod
    @abstractmethod
    def load_latest_parameters(filename: str) -> dict:
        pass


class Callback(ABC):
    """Abstract base class for callbacks such as `.CSVSummary`.

    .. seealso:: :ref:`usage/step3:Custom callbacks`
    """

    @abstractmethod
    def on_iteration_end(
        self, function_call: int, logs: Optional[Dict[str, Any]] = None
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
    >>> optimizer = Minuit2(
    ...     callback=CallbackList([TFSummary(), YAMLSummary("result.yml")])
    ... )
    """

    def __init__(self, callbacks: Iterable[Callback]) -> None:
        self.__callbacks: List[Callback] = list()
        for callback in callbacks:
            self.__callbacks.append(callback)

    def on_iteration_end(
        self, function_call: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        for callback in self.__callbacks:
            callback.on_iteration_end(function_call, logs)

    def on_function_call_end(self) -> None:
        for callback in self.__callbacks:
            callback.on_function_call_end()


class CSVSummary(Callback, Loadable):
    def __init__(self, filename: str, step_size: int = 10) -> None:
        """Log fit parameters and the estimator value to a CSV file."""
        self.__step_size = step_size
        self.__first_call = True
        self.__stream = open(filename, "w")
        _empty_file(self.__stream)

    def on_iteration_end(
        self, function_call: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        if logs is None:
            return
        if function_call % self.__step_size != 0:
            return
        output_dict = {
            "function_call": function_call,
            "time": logs["time"],
            "estimator_type": logs["estimator"]["type"],
            "estimator_value": logs["estimator"]["value"],
            **logs["parameters"],
        }
        data_frame = pd.DataFrame(output_dict, index=[function_call])
        data_frame.to_csv(
            self.__stream,
            mode="a",
            header=self.__first_call,
            index=False,
        )
        self.__first_call = False

    def on_function_call_end(self) -> None:
        self.__stream.close()

    @staticmethod
    def load_latest_parameters(filename: str) -> dict:
        fit_traceback = pd.read_csv(filename)
        parameter_traceback = fit_traceback[fit_traceback.columns[4:]]
        parameter_names = parameter_traceback.columns
        latest_parameter_values = parameter_traceback.iloc[-1]
        return dict(zip(parameter_names, latest_parameter_values))


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
        output_dir = logdir + "/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        if subdir is not None:
            output_dir += "/" + subdir
        self.__file_writer = tf.summary.create_file_writer(output_dir)
        self.__file_writer.set_as_default()
        self.__step_size = step_size

    def on_iteration_end(
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

    def on_function_call_end(self) -> None:
        self.__file_writer.close()


class YAMLSummary(Callback, Loadable):
    def __init__(self, filename: str, step_size: int = 10) -> None:
        """Log fit parameters and the estimator value to a `tf.summary`.

        The logs can be viewed with `TensorBoard
        <https://www.tensorflow.org/tensorboard>`_ via:

        .. code-block:: shell

            tensorboard --logdir logs
        """
        self.__step_size = step_size
        self.__stream = open(filename, "w")

    def on_iteration_end(
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

    def on_function_call_end(self) -> None:
        self.__stream.close()

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
