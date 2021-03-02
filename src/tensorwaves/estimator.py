"""Defines estimators which estimate a model's ability to represent the data.

All estimators have to implement the `~.interfaces.Estimator` interface.
"""
from typing import Callable, Dict, List, Union

from tensorwaves.interfaces import Estimator
from tensorwaves.physics.amplitude import (
    SympyModel,
    get_backend_modules,
    lambdify,
)


def gradient_creator(
    function: Callable[[Dict[str, Union[float, complex]]], float],
    backend: Union[str, tuple, dict],
) -> Callable[
    [Dict[str, Union[float, complex]]], Dict[str, Union[float, complex]]
]:
    # pylint: disable=import-outside-toplevel
    def not_implemented(
        parameters: Dict[str, Union[float, complex]]
    ) -> Dict[str, Union[float, complex]]:
        raise NotImplementedError("Gradient not implemented.")

    if isinstance(backend, str) and backend == "jax":
        import jax
        from jax.config import config

        config.update("jax_enable_x64", True)

        return jax.grad(function)

    return not_implemented


class SympyUnbinnedNLL(  # pylint: disable=too-many-instance-attributes
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
        backend: Union[str, tuple, dict] = "numpy",
    ) -> None:
        self.__gradient = gradient_creator(self.__call__, backend)
        backend_modules = get_backend_modules(backend)

        self.__parameters: Dict[str, Union[float, complex]] = {
            k.name: v for k, v in model.parameters.items()
        }

        model_expr = model.expression.doit()

        self.__bare_model = lambdify(
            tuple(model_expr.free_symbols),
            model_expr,
            backend=backend,
        )

        def find_function_in_backend(name: str) -> Callable:
            if isinstance(backend_modules, dict) and name in backend_modules:
                return backend_modules[name]
            if isinstance(backend_modules, (tuple, list)):
                for module in backend_modules:
                    if name in module.__dict__:
                        return module.__dict__[name]
            raise ValueError(f"Could not find function {name} in backend")

        self.__mean_function = find_function_in_backend("mean")
        self.__sum_function = find_function_in_backend("sum")
        self.__log_function = find_function_in_backend("log")

        self.__phsp_volume = phsp_volume

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

    def __call__(self, parameters: Dict[str, Union[float, complex]]) -> float:
        self.__update_parameters(parameters)

        bare_intensities = self.__bare_model(*self.__data_args)
        normalization_factor = 1.0 / (
            self.__phsp_volume
            * self.__mean_function(self.__bare_model(*self.__phsp_args))
        )
        likelihoods = normalization_factor * bare_intensities
        return -self.__sum_function(self.__log_function(likelihoods))

    def __update_parameters(
        self, parameters: Dict[str, Union[float, complex]]
    ) -> None:
        for par_name, value in parameters.items():
            if par_name in self.__parameter_index_mapping:
                index = self.__parameter_index_mapping[par_name]
                self.__data_args[index] = value
                self.__phsp_args[index] = value

    @property
    def parameters(self) -> List[str]:
        return list(self.__parameters.keys())

    def gradient(
        self, parameters: Dict[str, Union[float, complex]]
    ) -> Dict[str, Union[float, complex]]:
        return self.__gradient(parameters)
