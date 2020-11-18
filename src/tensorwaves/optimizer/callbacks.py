"""Collection of loggers that can be inserted into an optimizer as callback."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import IO, Iterable, List, Optional

import tensorflow as tf
import yaml
from tqdm import tqdm

from tensorwaves.interfaces import Estimator


class Callback(ABC):
    @abstractmethod
    def __call__(self, parameters: dict, estimator_value: float) -> None:
        pass

    @abstractmethod
    def finalize(self) -> None:
        pass


class ProgressBar(Callback):
    def __init__(self) -> None:
        self.__progress_bar = tqdm()

    def __call__(self, parameters: dict, estimator_value: float) -> None:
        self.__progress_bar.set_postfix({"estimator": estimator_value})
        self.__progress_bar.update()

    def finalize(self) -> None:
        self.__progress_bar.close()


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

    def __call__(self, parameters: dict, estimator_value: float) -> None:
        self.__function_call += 1
        if self.__function_call % self.__step_size != 0:
            return
        output_dict = {
            "Time": datetime.now(),
            "Iteration": self.__function_call,
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

    def finalize(self) -> None:
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

    def __call__(self, parameters: dict, estimator_value: float) -> None:
        self.__function_call += 1
        if self.__function_call % self.__step_size != 0:
            return
        for par_name, value in parameters.items():
            tf.summary.scalar(par_name, value, step=self.__function_call)
        tf.summary.scalar(
            "estimator", estimator_value, step=self.__function_call
        )
        self.__file_writer.flush()

    def finalize(self) -> None:
        self.__file_writer.close()


class CallbackList(Callback):
    """Class for combining `Callback` s.

    Combine different `Callback` classes in to a chain as follows:

    >>> from tensorwaves.optimizer.callbacks import (
    ...     CallbackList, ProgressBar, TFSummary
    ... )
    >>> from tensorwaves.optimizer.minuit import Minuit2
    >>> optimizer = Minuit2(callback=CallbackList([ProgressBar(), TFSummary()]))
    """

    def __init__(self, callbacks: Iterable[Callback]) -> None:
        self.__callbacks: List[Callback] = list()
        for callback in callbacks:
            self.__callbacks.append(callback)

    def __call__(self, parameters: dict, estimator_value: float) -> None:
        for callback in self.__callbacks:
            callback(parameters, estimator_value)

    def finalize(self) -> None:
        for callback in self.__callbacks:
            callback.finalize()


class _IncreasedIndent(yaml.Dumper):
    # pylint: disable=too-many-ancestors
    def increase_indent(self, flow=False, indentless=False):  # type: ignore
        return super().increase_indent(flow, False)


def _empty_file(stream: IO) -> None:
    stream.seek(0)
    stream.truncate()
