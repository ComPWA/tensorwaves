"""Defines estimators which estimate a model's ability to represent the data.

All estimators have to implement the `~.interfaces.Estimator` interface.
"""
from typing import Dict

import numpy as np
import tensorflow as tf

from tensorwaves.interfaces import Estimator, Function


class _NormalizedFunction(Function):
    def __init__(
        self,
        unnormalized_function: Function,
        norm_dataset: dict,
        norm_volume: float = 1.0,
    ) -> None:
        self._model = unnormalized_function
        # it is crucial to convert the input data to tensors
        # otherwise the memory footprint can increase dramatically
        self._norm_dataset = {
            x: tf.constant(y) for x, y in norm_dataset.items()
        }
        self._norm_volume = norm_volume

    def __call__(self, dataset: dict) -> tf.Tensor:
        normalization = tf.multiply(
            self._norm_volume,
            tf.reduce_mean(self._model(self._norm_dataset)),
        )
        return tf.divide(self._model(dataset), normalization)

    @property
    def parameters(self) -> Dict[str, tf.Variable]:
        return self._model.parameters

    def update_parameters(self, new_parameters: dict) -> None:
        self._model.update_parameters(new_parameters)


class UnbinnedNLL(Estimator):
    """Unbinned negative log likelihood estimator.

    Args:
        model: A model that should be compared to the dataset.
        dataset: The dataset used for the comparison. The model has to be
            evaluateable with this dataset.
        phsp_set: A phase space dataset, which is used for the normalization.
            The model has to be evaluateable with this dataset. When correcting
            for the detector efficiency use a phase space sample, that passed
            the detector reconstruction.

    """

    def __init__(self, model: Function, dataset: dict, phsp_set: dict) -> None:
        if phsp_set and len(phsp_set) > 0:
            self.__model: Function = _NormalizedFunction(model, phsp_set)
        else:
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
