# cspell:ignore scipy, asarray, nquad, yerr, ylim, ylabel, histogramdd

from functools import reduce
from math import cos, sqrt
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pytest
import scipy.integrate as integrate
from expertsystem.io import load_amplitude_model
from matplotlib import pyplot as plt

from tensorwaves.data.generate import generate_data
from tensorwaves.physics.helicity_formalism.amplitude import IntensityBuilder
from tensorwaves.physics.helicity_formalism.kinematics import (
    HelicityKinematics,
)


class Histogram:
    def __init__(
        self,
        var_name: str,
        bin_edges: Sequence[Sequence[float]],
        bin_contents: Sequence[float],
        bin_errors: Sequence[float],
        **mpl_kwargs: Any,
    ) -> None:
        self.var_name = var_name
        self.bin_edges = bin_edges
        self.bin_contents = bin_contents
        self.bin_errors = bin_errors
        self.mpl_kwargs = mpl_kwargs


def __chisquare(
    values: Sequence[float], errors: Sequence[float], expected: Sequence[float]
) -> float:
    return np.sum(
        [((x[0] - x[1]) / x[2]) ** 2 for x in zip(values, expected, errors)]
    )


@pytest.fixture(scope="module")
def chisquare_test() -> Callable[[Histogram, Callable], None]:
    def __chisquare_test(histogram: Histogram, func: Callable) -> None:
        function_hist = __function_to_histogram(func, histogram)
        function_hist = __scale_to_other_histogram(function_hist, histogram)
        degrees_of_freedom = (
            reduce(
                (lambda x, y: x * y), np.asarray(histogram.bin_contents).shape
            )
            - 1
        )

        redchi2 = (
            __chisquare(
                histogram.bin_contents,
                histogram.bin_errors,
                function_hist.bin_contents,
            )
            / degrees_of_freedom
        )
        error = sqrt(2 / degrees_of_freedom)  # for large degrees of freedom

        assert abs(redchi2 - 1.0) < 2.0 * error

    return __chisquare_test


@pytest.fixture(scope="module")
def residual_test() -> Callable[[Histogram, Callable], None]:
    def __residual_test(histogram: Histogram, func: Callable) -> None:
        function_hist = __function_to_histogram(func, histogram)
        function_hist = __scale_to_other_histogram(function_hist, histogram)

        residuals = [
            (x[0] - x[1]) / x[2]
            for x in zip(
                histogram.bin_contents,
                function_hist.bin_contents,
                histogram.bin_errors,
            )
        ]

        mean_error = 0.1  # should be calculated from the residuals
        rms_error = 0.1  # should be calculated from the residuals
        assert abs(np.mean(residuals)) < mean_error
        assert abs(np.sqrt(np.mean(np.square(residuals))) - 1.0) < rms_error

    return __residual_test


def __function_to_histogram(func: Callable, histogram: Histogram) -> Histogram:
    bin_edges = histogram.bin_edges

    integrals, errors = __integrate_within_bins(
        func, list(zip(bin_edges[0][:-1], bin_edges[0][1:]))
    )
    return Histogram(
        histogram.var_name,
        bin_edges,
        integrals,
        errors,
    )


def __integrate_within_bins(
    func: Callable, integration_ranges: Sequence[Tuple[float, float]]
) -> Tuple[Sequence[float], Sequence[float]]:
    results = [integrate.nquad(func, [x]) for x in integration_ranges]
    return ([x[0] for x in results], [x[1] for x in results])


def __scale_to_other_histogram(
    histogram: Histogram, histogram_reference: Histogram
) -> Histogram:
    normalization = np.sum(histogram_reference.bin_contents) / np.sum(
        histogram.bin_contents
    )

    new_bin_contents = np.multiply(normalization, histogram.bin_contents)
    new_bin_errors = [np.sqrt(normalization) * x for x in histogram.bin_errors]
    return Histogram(
        histogram.var_name,
        histogram.bin_edges,
        new_bin_contents,
        bin_errors=new_bin_errors,
    )


def __to_cosine(
    datarecord: np.ndarray, column_name: str
) -> Tuple[np.array, str]:
    return (
        [cos(x) for x in datarecord[column_name]],
        "cos" + column_name,
    )


def __make_histogram(
    var_name: str,
    values: np.array,
    weights: Optional[np.array] = None,
    bins: int = 50,
    **kwargs: Any,
) -> Histogram:
    bin_content, bin_edges = np.histogramdd(values, bins=bins, weights=weights)
    if len(bin_content.shape) == 1:
        errs = [np.sqrt(x) if x > 0 else 1 for x in bin_content]
    elif len(bin_content.shape) == 2:
        errs = [
            [np.sqrt(x) if x > 0 else 1 for x in row] for row in bin_content
        ]
    return Histogram(var_name, bin_edges, bin_content, errs, **kwargs)


def __plot_distributions_1d(
    histograms: Dict[str, Histogram],
    use_bin_centers: bool = True,
    **kwargs: Any,
) -> None:
    plt.clf()
    var_name = ""
    for name, histogram in histograms.items():
        bincenters = histogram.bin_edges
        if use_bin_centers:
            bincenters = 0.5 * (
                np.array(histogram.bin_edges[0][1:])
                + np.array(histogram.bin_edges[0][:-1])
            )
        plt.errorbar(
            bincenters,
            histogram.bin_contents,
            yerr=histogram.bin_errors,
            label=name,
            **(histogram.mpl_kwargs),
        )
        if var_name == "":
            var_name = histogram.var_name

    if plt.ylim()[0] > 0.0:
        plt.ylim(bottom=0.0)
    axis = plt.gca()
    if "x_title" in kwargs:
        axis.set_xlabel(kwargs["x_title"])
    else:
        axis.set_xlabel(var_name)
    axis.set_ylabel("")
    axis.legend()
    plt.tight_layout()
    plt.savefig(var_name + ".png", bbox_inches="tight")


@pytest.fixture(scope="module")
def generate_dataset() -> Callable[[str, int], np.ndarray]:
    def dataset_generator(model_filename: str, events: int) -> np.ndarray:
        model = load_amplitude_model(model_filename)

        kinematics = HelicityKinematics.from_model(model)
        part_list = model.particles

        builder = IntensityBuilder(part_list, kinematics)
        intensity = builder.create_intensity(model)

        sample = generate_data(events, kinematics, intensity)

        return kinematics.convert(sample)

    return dataset_generator


@pytest.fixture(scope="module")
def test_angular_distribution() -> Callable:
    def test_distribution(
        dataset: np.ndarray,
        variable_name: str,
        expected_distribution_function: Callable,
        test_function: Callable[[Histogram, Callable], None],
        bins: int = 120,
        make_plots: bool = False,
    ) -> None:
        if "theta" in variable_name and "cos" not in variable_name:
            var_data, var_name = __to_cosine(dataset, variable_name)
        else:
            var_data = dataset[variable_name]
            var_name = variable_name

        data_hist = __make_histogram(
            var_name,
            var_data,
            bins=bins,
            fmt="o",
        )

        if make_plots:
            function_hist = __function_to_histogram(
                expected_distribution_function, data_hist
            )
            function_hist = __scale_to_other_histogram(
                function_hist, data_hist
            )
            function_hist.mpl_kwargs = {"fmt": "-"}

            hist_bundle = {"data": data_hist, "theory": function_hist}
            __plot_distributions_1d(hist_bundle, x_title=var_name)

        test_function(data_hist, expected_distribution_function)

    return test_distribution
