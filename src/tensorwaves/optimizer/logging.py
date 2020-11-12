"""Logging of variables during optimization.

This module contains a logging decorator `.tf_file_logging`, that can be
used to decorated estimator updates during fitting. The logs can be viewed
with Tensorboard via

.. code-block:: bash

    tensorboard --logdir logs

"""

from datetime import datetime
from typing import Callable

import tensorflow as tf


def tf_file_logging(
    iterations: int = 2, filename: str = "parameters", logdir: str = "logs"
) -> Callable[[Callable], Callable]:
    file_writer = tf.summary.create_file_writer(
        logdir
        + "/"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
        + "/"
        + filename
    )
    file_writer.set_as_default()

    def write(
        estimator_value: float, parameters: dict, iteration: int
    ) -> None:
        for par_name, value in parameters.items():
            tf.summary.scalar(par_name, value, step=iteration)
        tf.summary.scalar("estimator", estimator_value, step=iteration)
        file_writer.flush()

    def decorator(
        function: Callable[[dict], float]
    ) -> Callable[[dict], float]:
        function_calls = 0

        def wrapper(parameters: dict) -> float:
            nonlocal function_calls
            function_calls += 1
            estimator_value = function(parameters)
            if function_calls % iterations == 0:
                write(estimator_value, parameters, function_calls)
            return estimator_value

        return wrapper

    return decorator
