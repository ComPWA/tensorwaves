# cspell:ignore asarray dphi epem histogramdd isclass nquad scipy yerr ylabel
# cspell:ignore ylim
# pylint: disable=import-outside-toplevel,no-self-use,redefined-outer-name

import os
from functools import reduce
from math import cos, sqrt
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import expertsystem as es
import numpy as np
import pytest
import scipy.integrate as integrate
from matplotlib import pyplot as plt

from tensorwaves.data.generate import (
    TFUniformRealNumberGenerator,
    generate_data,
)
from tensorwaves.physics.helicity_formalism.amplitude import IntensityBuilder
from tensorwaves.physics.helicity_formalism.kinematics import (
    HelicityKinematics,
)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


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


def chisquare_test(histogram: Histogram, func: Callable) -> None:
    function_hist = __function_to_histogram(func, histogram)
    function_hist = __scale_to_other_histogram(function_hist, histogram)
    degrees_of_freedom = (
        reduce((lambda x, y: x * y), np.asarray(histogram.bin_contents).shape)
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
    error = sqrt(
        2 / degrees_of_freedom
    )  # accurate for large degrees of freedom and gaussian errors

    assert abs(redchi2 - 1.0) < 3.0 * error


def residual_test(histogram: Histogram, func: Callable) -> None:
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
    _n = len(histogram.bin_contents)
    mean_error = sqrt(_n)
    sample_variance = np.sum(np.square(residuals)) / (_n - 1)
    sample_std_dev_error = sqrt(
        sample_variance / (2.0 * (_n - 1))
    )  # only true for gaussian distribution
    assert abs(np.mean(residuals)) < mean_error
    assert abs(sqrt(sample_variance) - 1.0) < 3.0 * sample_std_dev_error


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


def generate_dataset(model_filename: str, events: int) -> np.ndarray:
    model = es.io.load_amplitude_model(model_filename)

    kinematics = HelicityKinematics.from_model(model)
    part_list = model.particles

    builder = IntensityBuilder(part_list, kinematics)
    intensity = builder.create_intensity(model)

    rng = TFUniformRealNumberGenerator(seed=0)
    sample = generate_data(events, kinematics, intensity, random_generator=rng)

    return kinematics.convert(sample)


def verify_angular_distribution(
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
        function_hist = __scale_to_other_histogram(function_hist, data_hist)
        function_hist.mpl_kwargs = {"fmt": "-"}

        hist_bundle = {"data": data_hist, "theory": function_hist}
        __plot_distributions_1d(hist_bundle, x_title=var_name)

    test_function(data_hist, expected_distribution_function)


class TestEpemToDmD0Pip:
    # Use this function to reproduce the model file.
    # Note the normalization part has been removed!
    def generate_model(self) -> None:
        epem = es.particle.Particle(
            name="EpEm",
            pid=12345678,
            mass=4.36,
            spin=1.0,
            parity=es.particle.Parity(-1),
            c_parity=es.particle.Parity(-1),
        )
        particles = es.io.load_pdg()
        particles.add(epem)

        result = es.generate_transitions(
            initial_state=[("EpEm", [-1])],
            final_state=[("D0", [0]), ("D-", [0]), ("pi+", [0])],
            allowed_intermediate_particles=["D(2)*(2460)+"],
            allowed_interaction_types="em",
            particles=particles,
        )

        amplitude_model = es.generate_amplitudes(result)
        amplitude_model.dynamics.set_non_dynamic("D(2)*(2460)+")
        es.io.write(
            amplitude_model, f"{SCRIPT_DIR}/{self.__class__.__name__}.yml"
        )

    # Use this function to reproduce the theoretical predictions.
    @staticmethod
    def calc_distributions() -> List[Tuple[str, Any]]:
        import sympy
        from sympy.abc import symbols
        from sympy.physics.quantum.spin import WignerD

        def calculate_integral(
            intensity: Any,
            integration_variables: List[Any],
            jacobi_determinant: Optional[Any] = None,
        ) -> Any:
            if jacobi_determinant is None:
                for int_var in integration_variables:
                    if "theta" in int_var.name:
                        intensity *= sympy.sin(int_var)
            else:
                intensity *= jacobi_determinant
            return sympy.trigsimp(
                sympy.re(
                    sympy.integrate(
                        intensity,
                        *(
                            (x, -sympy.pi, sympy.pi)
                            if "phi" in x.name
                            else (x, 0, sympy.pi)
                            for x in integration_variables
                        ),
                    )
                )
            ).doit()

        theta1, phi1, theta2, phi2, dphi = symbols(
            "theta1,phi1,theta2,phi2,dphi", real=True
        )

        amp = (
            WignerD(1, -1, 1, -phi1, theta1, phi1)
            * WignerD(2, 1, 0, -phi2, theta2, phi2)
            - 1
            * WignerD(1, -1, -1, -phi1, theta1, phi1)
            * WignerD(2, -1, 0, -phi2, theta2, phi2)
        ).doit()

        intensity = sympy.simplify(
            (amp * sympy.conjugate(amp)).expand(complex=True)
        )
        intensity = sympy.simplify(intensity.replace(phi2, dphi + phi1))
        assert sympy.im(intensity) == 0

        all_variables = [theta1, phi1, theta2, dphi]
        return [
            (
                f"{var.name} dependency:",
                calculate_integral(
                    intensity,
                    all_variables[0:i] + all_variables[i + 1 :],
                ),
            )
            for i, var in enumerate(all_variables)
        ]

    @pytest.fixture(scope="module")
    def intensity_dataset(self) -> np.ndarray:
        return generate_dataset(
            model_filename=f"{SCRIPT_DIR}/{self.__class__.__name__}.yml",
            events=50000,
        )

    @pytest.mark.parametrize(
        "angular_variable, expected_distribution_function",  # type: ignore
        [
            (  # x = cos(theta) distribution from epem decay
                "theta_3+4_2",
                lambda x: 1 + x * x,
            ),
            (  # phi distribution of the epem decay
                "phi_3+4_2",
                lambda x: 1,
            ),
            (  # x = cos(theta') distribution from D2*
                "theta_3_4_vs_2",
                lambda x: 1 - (2 * x * x - 1) ** 2,
            ),
            (  # phi' distribution of the D2* decay
                "phi_3_4_vs_2",
                lambda phi: 2 + cos(2 * phi),
            ),
            # ( # 2d distribution of the D2* decay
            #   ['theta_3_4_vs_2', 'phi_3_4_vs_2'],
            #   lambda x, phi: (1 - x**2) * (x**2) * (2 + cos(2 * phi)),
            # )
        ],  # type: ignore
    )
    def test_distributions_reduced_chi2(
        self,
        angular_variable: str,
        expected_distribution_function: Callable,
        intensity_dataset,
    ) -> None:

        verify_angular_distribution(
            intensity_dataset,
            angular_variable,
            expected_distribution_function,
            chisquare_test,
            bins=180,
            make_plots=True,
        )


class TestD1ToD0PiPi:
    # Use this function to reproduce the model file.
    # Note the normalization part has been removed!
    def generate_model(self) -> None:
        result = es.generate_transitions(
            initial_state=[("D(1)(2420)0", [-1])],
            final_state=[("D0", [0]), ("pi-", [0]), ("pi+", [0])],
            allowed_intermediate_particles=["D*"],
            allowed_interaction_types="strong",
        )
        amplitude_model = es.generate_amplitudes(result)
        amplitude_model.dynamics.set_non_dynamic("D*(2010)+")
        es.io.write(
            amplitude_model, f"{SCRIPT_DIR}/{self.__class__.__name__}.yml"
        )

    # Use this function to reproduce the theoretical predictions.
    @staticmethod
    def calc_distributions() -> List[Tuple[str, Any]]:
        import sympy
        from sympy.abc import symbols
        from sympy.physics.quantum.spin import WignerD

        def calculate_integral(
            intensity: Any,
            integration_variables: List[Any],
            jacobi_determinant: Optional[Any] = None,
        ) -> Any:
            if jacobi_determinant is None:
                for int_var in integration_variables:
                    if "theta" in int_var.name:
                        intensity *= sympy.sin(int_var)
            else:
                intensity *= jacobi_determinant
            return sympy.trigsimp(
                sympy.re(
                    sympy.integrate(
                        intensity,
                        *(
                            (x, -sympy.pi, sympy.pi)
                            if "phi" in x.name
                            else (x, 0, sympy.pi)
                            for x in integration_variables
                        ),
                    )
                )
            ).doit()

        theta1, phi1, theta2, phi2 = symbols(
            "theta1,phi1,theta2,phi2", real=True
        )

        # The phi1 dependency vanishes completely, hence phi2 can be seen as the
        # difference between the two phi angles.
        amp = (
            WignerD(1, -1, -1, -phi1, theta1, 0)
            * WignerD(1, -1, 0, -phi2, theta2, 0)
            + 0.5
            * WignerD(1, -1, 0, -phi1, theta1, 0)
            * WignerD(1, 0, 0, -phi2, theta2, 0)
            + WignerD(1, -1, 1, -phi1, theta1, 0)
            * WignerD(1, 1, 0, -phi2, theta2, 0)
        ).doit()

        intensity = sympy.simplify(
            (amp * sympy.conjugate(amp)).expand(complex=True)
        )

        assert sympy.im(intensity) == 0

        all_variables = [theta1, phi1, theta2, phi2]
        return [
            (
                f"{var.name} dependency:",
                calculate_integral(
                    intensity,
                    all_variables[0:i] + all_variables[i + 1 :],
                ),
            )
            for i, var in enumerate(all_variables)
        ]

    @pytest.fixture(scope="module")
    def intensity_dataset(self) -> np.ndarray:
        return generate_dataset(
            model_filename=f"{SCRIPT_DIR}/{self.__class__.__name__}.yml",
            events=30000,
        )

    @pytest.mark.parametrize(
        "angular_variable, expected_distribution_function",  # type: ignore
        [
            (  # x = cos(theta) distribution from D1 decay
                "theta_3+4_2",
                lambda x: 1.25 + 0.75 * x * x,
            ),
            (  # x = cos(theta') distribution from D*
                "theta_3_4_vs_2",
                lambda x: 1 - 0.75 * x * x,
            ),
            (  # phi distribution of the D* decay
                "phi_3_4_vs_2",
                lambda x: 1 - 1 / 2.25 * cos(2 * x),
            ),
        ],  # type: ignore
    )
    def test_distributions_reduced_chi2(
        self,
        angular_variable: str,
        expected_distribution_function: Callable,
        intensity_dataset,
    ) -> None:

        verify_angular_distribution(
            intensity_dataset,
            angular_variable,
            expected_distribution_function,
            chisquare_test,
            bins=120,
            make_plots=False,
        )

    @pytest.mark.parametrize(
        "angular_variable, expected_distribution_function",  # type: ignore
        [
            (  # x = cos(theta) distribution from D1 decay
                "theta_3+4_2",
                lambda x: 1.25 + 0.75 * x * x,
            ),
            (  # x = cos(theta') distribution from D*
                "theta_3_4_vs_2",
                lambda x: 1 - 0.75 * x * x,
            ),
            (  # phi distribution of the D* decay
                "phi_3_4_vs_2",
                lambda x: 1 - 1 / 2.25 * cos(2 * x),
            ),
        ],  # type: ignore
    )
    def test_distributions_residuals(
        self,
        angular_variable: str,
        expected_distribution_function: Callable,
        intensity_dataset,
    ) -> None:

        verify_angular_distribution(
            intensity_dataset,
            angular_variable,
            expected_distribution_function,
            residual_test,
            bins=120,
            make_plots=False,
        )


def generate_models():
    import inspect
    import sys

    for _, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) and hasattr(obj, "generate_model"):
            print(f"Genenerating model for {obj.__name__}")
            instance = obj()
            instance.generate_model()


if __name__ == "__main__":
    generate_models()
