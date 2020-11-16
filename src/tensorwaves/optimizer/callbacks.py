"""Collection of loggers that can be inserted into an optimizer as callback."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Iterable, List, Optional

import tensorflow as tf
from tqdm import tqdm


class Callback(ABC):
    @abstractmethod
    def __call__(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def finalize(self) -> None:
        pass


class ProgressBar:
    def __init__(self) -> None:
        self.__progress_bar = tqdm()

    def __call__(self, estimator_value: float, **kwargs: Any) -> None:
        self.__progress_bar.set_postfix({"estimator": estimator_value})
        self.__progress_bar.update()

    def finalize(self) -> None:
        self.__progress_bar.close()


class TFSummary:
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
        self.__iteration = 0
        self.__step_size = step_size

    def __call__(
        self,
        parameters: dict,
        estimator_value: float,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.__iteration += 1
        if self.__iteration % self.__step_size != 0:
            return
        for par_name, value in parameters.items():
            tf.summary.scalar(par_name, value, step=self.__iteration)
        tf.summary.scalar("estimator", estimator_value, step=self.__iteration)
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

    def __call__(self, **kwargs: Any) -> None:
        for callback in self.__callbacks:
            callback(**kwargs)

    def finalize(self) -> None:
        for callback in self.__callbacks:
            callback.finalize()
