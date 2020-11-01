"""Defines estimators which estimate a model's ability to represent the data.

All estimators have to implement the `~.interfaces.Estimator` interface.
"""
import numpy as np
import tensorflow as tf

from tensorwaves.interfaces import Estimator, Function


class UnbinnedNLL(Estimator):
    """Unbinned negative log likelihood estimator.

    Args:
        model: A model that should be compared to the dataset.
        dataset: The dataset used for the comparison. The model has to be
            evaluateable with this dataset.

    """

    def __init__(self, model: Function, dataset: dict) -> None:
        self.__model = model
        self.__dataset = dataset

    def __call__(self) -> float:
        props = self.__model(self.__dataset)
        logs = tf.math.log(props)
        log_lh = tf.reduce_sum(logs)
        return -log_lh.numpy()

    def gradient(self) -> np.ndarray:
        raise NotImplementedError("Gradient not implemented.")

    @property
    def parameters(self) -> dict:
        return self.__model.parameters

    def update_parameters(self, new_parameters: dict) -> None:
        self.__model.update_parameters(new_parameters)
