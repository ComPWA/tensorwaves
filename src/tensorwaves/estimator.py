"""Defines estimators which estimate a model's ability to represent the data.

All estimators have to implement the `~.interfaces.Estimator` interface.
"""
from typing import Dict, List, Union

import numpy as np
import sympy
import tensorflow as tf

from tensorwaves.interfaces import Estimator, Function
from tensorwaves.physics.amplitude import SympyModel


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

    def __call__(self, new_parameters: Dict[str, float]) -> float:
        self.__model.update_parameters(new_parameters)
        props = self.__model(self.__dataset)
        logs = tf.math.log(props)
        log_lh = tf.reduce_sum(logs)
        return -log_lh.numpy()

    def gradient(self) -> np.ndarray:
        raise NotImplementedError("Gradient not implemented.")

    @property
    def parameters(self) -> List[str]:
        return list(self.__model.parameters.keys())


class SympyUnbinnedNLL(  # pylint: disable= too-many-instance-attributes
    Estimator
):
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

    def __init__(
        self,
        model: SympyModel,
        dataset: dict,
        phsp_dataset: dict,
        phsp_volume: float = 1.0,
        backend: Union[str, dict] = "numpy",
    ) -> None:
        model_expr = model.expression.doit()

        self.__bare_model = sympy.lambdify(
            tuple(model_expr.free_symbols),
            model_expr,
            modules=backend,
        )

        # This part would have to be modified to the various backends
        self.__mean_function = np.mean
        self.__sum_function = np.sum
        self.__log_function = np.log

        self.__phsp_volume = phsp_volume
        self.__parameters: Dict[str, float] = {
            k.name: v for k, v in model.parameters.items()
        }

        self.__data_args = []
        self.__phsp_args = []
        self.__parameter_index_mapping: Dict[str, int] = {}

        for i, var_name in enumerate(
            tuple(x.name for x in model_expr.free_symbols)
        ):
            if var_name in dataset and var_name in phsp_dataset:
                self.__data_args.append(dataset[var_name])
                self.__phsp_args.append(phsp_dataset[var_name])
            elif var_name in dataset:
                raise ValueError(
                    f"Datasets do not match! {var_name} exists in dataset but "
                    "not in phase space dataset."
                )
            elif var_name in phsp_dataset:
                raise ValueError(
                    f"Datasets do not match! {var_name} exists in phase space "
                    "dataset but not in dataset."
                )
            else:
                self.__data_args.append(self.__parameters[var_name])
                self.__phsp_args.append(self.__parameters[var_name])
                self.__parameter_index_mapping[var_name] = i

    def __call__(self, parameters: Dict[str, float]) -> float:
        self.__update_parameters(parameters)

        bare_intensities = self.__bare_model(*self.__data_args)
        normalization_factor = 1.0 / (
            self.__phsp_volume
            * self.__mean_function(self.__bare_model(*self.__phsp_args))
        )
        likelihoods = normalization_factor * bare_intensities
        return -self.__sum_function(self.__log_function(likelihoods))

    def __update_parameters(self, parameters: Dict[str, float]) -> None:
        for par_name, value in parameters.items():
            if par_name in self.__parameter_index_mapping:
                index = self.__parameter_index_mapping[par_name]
                self.__data_args[index] = value
                self.__phsp_args[index] = value

    @property
    def parameters(self) -> List[str]:
        return list(self.__parameters.keys())
