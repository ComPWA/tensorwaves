"""Amplitude module for the helicity formalism.

Its responsibility is the construction of complicated helicity formalism
amplitude models using a recipe (see `IntensityBuilder`). These models are
encapsulated in an `IntensityTF` class, which can be evaluated as a regular
callable.
"""

import logging
from collections import namedtuple
from typing import Any, Callable, Dict, List, Tuple

import amplitf.interface as atfi  # type: ignore
from amplitf.dynamics import relativistic_breit_wigner  # type: ignore
from amplitf.kinematics import wigner_capital_d  # type: ignore

import numpy  # type: ignore

import tensorflow as tf  # type: ignore

from tensorwaves.interfaces import Function

from .kinematics import HelicityKinematics, SubSystem


class IntensityTF(Function):
    """Implementation of the `~.Function` interface using tensorflow.

    Initialize the intensity based on a tensorflow model.

    Args:
        tf_model: A callable with potential tensorflow code.
        parameters: The collection of parameters of the model.

    """

    def __init__(self, tf_model: Callable, parameters: Dict[str, tf.Variable]):
        self._model = tf_model
        self.__parameters = parameters

    def __call__(self, dataset: Dict[str, numpy.ndarray]) -> numpy.ndarray:
        """Evaluate the Intensity.

        Args:
            dataset: Contains all required kinematic variables.

        Returns:
            List of intensity values.
        """
        # it is crucial to convert the input data to tensors
        # otherwise the memory footprint can increase dramatically
        newdataset = {x: tf.constant(y) for x, y in dataset.items()}
        return self._model(newdataset).numpy()

    @property
    def parameters(self) -> dict:
        return {x: y.value().numpy() for x, y in self.__parameters.items()}

    def update_parameters(self, new_parameters: dict) -> None:
        for name, value in new_parameters.items():
            self.__parameters[name].assign(value)


class IntensityBuilder:
    """Builds Intensities from helicity formalism recipe files.

    Args:
        particles: Contains info of various particles.
        kinematics: A helicity kinematics instance. Note that this kinematics
            instance will be modified in the process.
        phsp_data: A phase space event collection, required if a normalization
            of the Intensity is performed.

    """

    def __init__(
        self,
        particles: dict,
        kinematics: HelicityKinematics,
        phsp_data: numpy.ndarray = numpy.array([]),
    ):
        self._particles = particles
        self._dynamics: Dict[str, Any] = {}
        self._kinematics = kinematics
        self._parameters: Dict[str, tf.Variable] = {}
        self._phsp_data = phsp_data
        self._registered_element_builders: Dict[str, Callable] = {
            "NormalizedIntensity": _create_normalized_intensity,
            "StrengthIntensity": _create_strength_intensity,
            "IncoherentIntensity": _create_incoherent_intensity,
            "CoherentIntensity": _create_coherent_intensity,
            "CoefficientAmplitude": _create_coefficient_amplitude,
            "SequentialAmplitude": _create_sequential_amplitude,
            "HelicityDecay": _create_helicity_decay,
        }
        self._registered_dynamics_builders: Dict[str, Callable] = {
            "nonDynamic": _create_no_dynamics,
            "relativisticBreitWigner": _create_relativistic_breit_wigner,
        }

    def create_intensity(self, recipe: dict) -> IntensityTF:
        """Create an `IntensityTF` instance based on a recipe.

        Args:
            recipe: Contains builder instructions. These recipe files can be
                generated via the expert system
                (see `~.expertsystem.amplitude.helicitydecay.HelicityAmplitudeGeneratorXML`)

        """
        self._dynamics = recipe["Dynamics"]
        if "Intensity" not in recipe:
            logging.error(
                "The recipe does not contain a Intensity. "
                "Please specify a recipe with an Intensity!"
            )
        self._initialize_parameters(recipe)
        return IntensityTF(
            self.create_element(recipe["Intensity"]), self._parameters
        )

    def create_element(self, recipe: dict) -> Callable:
        """Create a computation element from the recipe.

        The recipe can only contain names registered in the pool of known
        element builders.

        """
        element_class = recipe["Class"]
        logging.debug("creating %s", element_class)

        if element_class not in self._registered_element_builders:
            raise Exception(f"Unknown element {element_class}!")

        return self._registered_element_builders[element_class](
            self, recipe, kinematics=self._kinematics
        )

    def register_element_builder(
        self,
        element_name: str,
        builder: Callable[["IntensityBuilder", dict], Callable],
    ) -> None:
        """
        Register builder functions for a specific operation.

        Allows user to inject personalized building code which will be call once
        an element with corresponding name is found.

        """
        if element_name in self._registered_element_builders:
            logging.warning(
                "Overwriting previously defined builder for %s", element_name
            )
        self._registered_element_builders[element_name] = builder

    def create_dynamics(
        self,
        decaying_state_name: str,
        dynamics_properties: "DynamicsProperties",
    ) -> Callable:
        """Create a dynamics function callable."""
        dynamics_builder = self._get_dynamics_builder(decaying_state_name)
        return dynamics_builder(dynamics_properties)

    def register_dynamics_builder(
        self,
        dynamics_name: str,
        builder: Callable[[str, "DynamicsProperties"], Callable],
    ) -> None:
        """Register custom dynamics function builders."""
        if dynamics_name in self._registered_dynamics_builders:
            logging.warning(
                "Overwriting previously defined builder for %s", dynamics_name
            )
        self._registered_dynamics_builders[dynamics_name] = builder

    def _get_dynamics_builder(
        self, decaying_state_name: str
    ) -> Callable[["DynamicsProperties"], Callable]:
        if decaying_state_name not in self._dynamics:
            raise LookupError(
                f"Could not find dynamics for particle with name"
                f" {decaying_state_name}"
            )

        decay_dynamics = self._dynamics[decaying_state_name]
        dynamics_type = decay_dynamics["Type"]
        if dynamics_type not in self._registered_dynamics_builders:
            raise ValueError(
                f"Dynamics ({dynamics_type}) unknown. "
                f"Use one of the following: \n"
                f"{list(self._registered_dynamics_builders.keys())}"
            )
        return self._registered_dynamics_builders[dynamics_type]

    def get_particle_infos(self, particle_name: str) -> dict:
        """Obtain particle information identified by its name."""
        if particle_name not in self._particles:
            raise LookupError(
                f"Could not find particle with name {particle_name}"
            )
        return self._particles[particle_name]

    def register_parameter(self, name: str, value: float) -> tf.Variable:
        if name not in self._parameters:
            self._parameters[name] = tf.Variable(
                value, name=name, dtype=tf.float64
            )
        return self._parameters[name]

    def get_parameter(self, name: str) -> tf.Variable:
        if name not in self._parameters:
            raise Exception(
                "Parameter {name} not registered! Your recipe file is"
                " corrupted!"
            )

        return self._parameters[name]

    def _initialize_parameters(self, recipe: dict) -> None:
        for par in recipe["Parameters"]:
            self._parameters[par["Name"]] = tf.Variable(
                par["Value"], name=par["Name"], dtype=tf.float64
            )

    def get_normalization_data(self) -> Tuple[dict, float]:
        """Return phase space dataset and its volume."""
        if self._phsp_data.size == 0:
            raise Exception(
                "No phase space sample given! This is required for the "
                "normalization."
            )
        return (
            self._kinematics.convert(self._phsp_data),
            self._kinematics.phase_space_volume,
        )


class _NormalizedIntensity:
    def __init__(
        self,
        unnormalized_intensity: Callable,
        norm_dataset: dict,
        norm_volume: float = 1.0,
    ) -> None:
        self._model = unnormalized_intensity
        self._norm_dataset = norm_dataset
        self._norm_volume = norm_volume

    @tf.function
    def __call__(self, dataset: dict) -> tf.Tensor:
        normalization = tf.multiply(
            self._norm_volume, tf.reduce_mean(self._model(self._norm_dataset)),
        )
        return tf.divide(self._model(dataset), normalization)


def _create_normalized_intensity(
    builder: IntensityBuilder, recipe: dict, **_: dict
) -> Callable:
    model = builder.create_element(recipe["Intensity"])
    dataset, volume = builder.get_normalization_data()
    # its important to convert the dataset to tf tensors (memory footprint)
    dataset = {x: tf.constant(y) for x, y in dataset.items()}
    return _NormalizedIntensity(model, dataset, atfi.const(volume))


class _StrengthIntensity:
    def __init__(self, intensity: Callable, strength: tf.Variable) -> None:
        self._strength = strength
        self._intensity = intensity

    def __call__(self, dataset: dict) -> tf.Tensor:
        return self._strength * self._intensity(dataset)


def _create_strength_intensity(
    builder: IntensityBuilder, recipe: dict, **_: dict
) -> Callable:
    strength = builder.get_parameter(recipe["Strength"])
    intensity = builder.create_element(recipe["Intensity"])
    return _StrengthIntensity(intensity, strength)


class _IncoherentIntensity:
    def __init__(self, intensities: List[Callable]) -> None:
        self._intensities = intensities

    def __call__(self, dataset: dict) -> tf.Tensor:
        return tf.math.accumulate_n([y(dataset) for y in self._intensities])


def _create_incoherent_intensity(
    builder: IntensityBuilder, recipe: dict, **_: dict
) -> Callable:
    if not isinstance(recipe["Intensities"], list):
        raise Exception("Incoherent Intensity requires a list of intensities!")
    intensities = [builder.create_element(x) for x in recipe["Intensities"]]
    return _IncoherentIntensity(intensities)


class _CoherentIntensity:
    def __init__(self, amplitudes: List[Callable]) -> None:
        self._amps = amplitudes

    def __call__(self, dataset: dict) -> tf.Tensor:
        return tf.pow(
            tf.cast(
                tf.abs(tf.add_n([amp(dataset) for amp in self._amps])),
                dtype=tf.float64,
            ),
            tf.constant(2.0, dtype=tf.float64),
        )


def _create_coherent_intensity(
    builder: IntensityBuilder, recipe: dict, **_: dict
) -> Callable:
    if not isinstance(recipe["Amplitudes"], list):
        raise Exception("Coherent Intensity requires a list of amplitudes!")
    amps = [builder.create_element(x) for x in recipe["Amplitudes"]]
    return _CoherentIntensity(amps)


class _CoefficientAmplitude:
    def __init__(
        self, amplitude: Callable, mag: tf.Variable, phase: tf.Variable
    ):
        self._mag = mag
        self._phase = phase
        self._amp = amplitude

    def __call__(self, dataset: dict) -> tf.Tensor:
        coefficient = atfi.polar(self._mag, self._phase)
        return coefficient * self._amp(dataset)


def _create_coefficient_amplitude(
    builder: IntensityBuilder, recipe: dict, **_: dict
) -> Callable:
    if not isinstance(recipe, dict):
        raise ValueError("recipe corrupted!")

    pars = recipe["Parameters"]
    mag = builder.get_parameter(pars["Magnitude"])
    phase = builder.get_parameter(pars["Phase"])
    amp = builder.create_element(recipe["Amplitude"])
    return _CoefficientAmplitude(amp, mag, phase)


class _SequentialAmplitude:
    def __init__(self, amplitudes: List[Callable]) -> None:
        self._seq_amps = amplitudes

    def __call__(self, dataset: dict) -> tf.Tensor:
        seq_amp = atfi.complex(atfi.const(1.0), atfi.const(0.0))
        for amp in self._seq_amps:
            seq_amp = seq_amp * amp(dataset)
        return seq_amp


def _create_sequential_amplitude(
    builder: IntensityBuilder, recipe: dict, **_: dict
) -> Callable:
    if not isinstance(recipe["Amplitudes"], list):
        raise Exception("Sequential Amplitude requires a list of amplitudes!")
    amp_recipes = recipe["Amplitudes"]
    if len(amp_recipes) == 0:
        raise Exception(
            "Sequential Amplitude requires a non-empty list of amplitudes!"
        )
    return _SequentialAmplitude(
        [builder.create_element(x) for x in amp_recipes]
    )


_AngularProperties = namedtuple(
    "_AngularProperties", ["j", "m", "mprime", "theta_name", "phi_name"],
)

DynamicsProperties = namedtuple(
    "DynamicsProperties",
    [
        "orbit_angular_momentum",
        "resonance_mass",
        "resonance_width",
        "inv_mass_name",
        "inv_mass_name_prod1",
        "inv_mass_name_prod2",
    ],
)


class _RelativisticBreitWigner:
    def __init__(
        self, mass: tf.Variable, width: tf.Variable, inv_mass_name: str
    ) -> None:
        self._resonance_mass = mass
        self._resonance_width = width
        self._inv_mass_name = inv_mass_name

    def __call__(self, dataset: dict) -> tf.Tensor:
        return relativistic_breit_wigner(
            dataset[self._inv_mass_name],
            self._resonance_mass,
            self._resonance_width,
        )


def _create_relativistic_breit_wigner(
    dynamics_props: DynamicsProperties,
) -> Callable:
    return _RelativisticBreitWigner(
        dynamics_props.resonance_mass,
        dynamics_props.resonance_width,
        dynamics_props.inv_mass_name,
    )


class _NonDynamic:
    def __call__(self, dataset: dict) -> tf.Tensor:
        return tf.complex(
            tf.constant(1.0, dtype=tf.float64),
            tf.constant(0.0, dtype=tf.float64),
        )


def _create_no_dynamics(_: DynamicsProperties) -> Callable:
    return _NonDynamic()


class _HelicityDecay:
    def __init__(
        self, angular_params: "_AngularProperties", dynamics_function: Callable
    ) -> None:
        self._params = angular_params
        self._dynamics_function = dynamics_function

    def __call__(self, dataset: dict) -> tf.Tensor:
        return wigner_capital_d(
            dataset[self._params.phi_name],
            dataset[self._params.theta_name],
            0.0,
            2 * self._params.j,
            2 * self._params.m,
            2 * self._params.mprime,
        ) * self._dynamics_function(dataset)


def _create_helicity_decay(
    builder: IntensityBuilder, recipe: dict, **kwargs: dict
) -> Callable:
    if not isinstance(recipe, dict):
        raise Exception("Helicity Decay expects a dictionary recipe!")
    decaying_state = recipe["DecayParticle"]
    decay_products = recipe["DecayProducts"]["Particle"]

    def to_list(ids: Any) -> list:
        return ids if isinstance(ids, list) else [ids]

    # define the subsystem
    dec_prod_fs_ids = [x["FinalState"] for x in decay_products]
    dec_prod_fs_ids = [to_list(x) for x in dec_prod_fs_ids]

    recoil_fs_ids = []
    parent_recoil_fs_ids = []
    if "RecoilSystem" in recipe:
        if "RecoilFinalState" in recipe["RecoilSystem"]:
            recoil_fs_ids = to_list(recipe["RecoilSystem"]["RecoilFinalState"])
        if "ParentRecoilFinalState" in recipe["RecoilSystem"]:
            parent_recoil_fs_ids = to_list(
                recipe["RecoilSystem"]["ParentRecoilFinalState"]
            )

    kinematics: HelicityKinematics = kwargs["kinematics"]

    (inv_mass_name, theta_name, phi_name) = kinematics.register_subsystem(
        SubSystem(dec_prod_fs_ids, recoil_fs_ids, parent_recoil_fs_ids)
    )

    decaying_state_name = decaying_state["Name"]

    particle_infos = builder.get_particle_infos(decaying_state_name)

    j = particle_infos["QuantumNumbers"]["Spin"]

    angular_params = _AngularProperties(
        j=j,
        m=decaying_state["Helicity"],
        mprime=decay_products[0]["Helicity"] - decay_products[1]["Helicity"],
        theta_name=theta_name,
        phi_name=phi_name,
    )

    dynamics_props = DynamicsProperties(
        orbit_angular_momentum=j,
        resonance_mass=builder.register_parameter(
            "Mass_" + decaying_state_name, particle_infos["Mass"]["Value"],
        ),
        resonance_width=builder.register_parameter(
            "Width_" + decaying_state_name, particle_infos["Width"]["Value"],
        ),
        inv_mass_name=inv_mass_name,
        inv_mass_name_prod1=kinematics.register_invariant_mass(
            dec_prod_fs_ids[0]
        ),
        inv_mass_name_prod2=kinematics.register_invariant_mass(
            dec_prod_fs_ids[1]
        ),
    )

    dynamics = builder.create_dynamics(decaying_state_name, dynamics_props)

    return _HelicityDecay(angular_params, dynamics)
