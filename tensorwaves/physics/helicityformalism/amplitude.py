"""Amplitude module for the helicity formalism.

Its responsibility is the construction of complicated helicity formalism
amplitude models using a recipe (see `IntensityBuilder`). These models are
encapsulated in an `IntensityTF` class, which can be evaluated as a regular
callable.
"""

import logging
from typing import Any, Callable, Dict, Optional

import amplitf.interface as atfi
from amplitf.dynamics import relativistic_breit_wigner
from amplitf.kinematics import wigner_capital_d

import numpy

import tensorflow as tf

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
            self._create_element(recipe["Intensity"]), self._parameters
        )

    def _create_element(self, recipe: dict) -> Callable:
        element_class = recipe["Class"]
        logging.debug("creating %s", element_class)

        if element_class not in self._registered_element_builders:
            raise Exception(f"Unknown element {element_class}!")

        return self._registered_element_builders[element_class](self, recipe)

    def register_element_builder(
        self,
        element_name: str,
        builder: Callable[["IntensityBuilder", dict], Callable],
    ):
        if element_name in self._registered_element_builders:
            logging.warning(
                "Overwriting previously defined builder for %s", element_name
            )
        self._registered_element_builders[element_name] = builder

    def _register_parameter(self, name: str, value: float) -> tf.Variable:
        if name not in self._parameters:
            self._parameters[name] = tf.Variable(
                value, name=name, dtype=tf.float64
            )
        return self._parameters[name]

    def _get_parameter(self, name: str) -> tf.Variable:
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

    def _get_normalization_data(self) -> Optional[dict]:
        if self._phsp_data.size == 0:
            raise Exception(
                "No phase space sample given! This is required for the "
                "normalization."
            )
        return self._phsp_data


class _NormalizedIntensity:
    def __init__(self, unnormalized_intensity, norm_dataset, norm_volume=1.0):
        self._model = unnormalized_intensity
        self._norm_dataset = norm_dataset
        self._norm_volume = norm_volume

    @tf.function
    def __call__(self, dataset):
        normalization = tf.multiply(
            self._norm_volume, tf.reduce_mean(self._model(self._norm_dataset)),
        )
        return tf.divide(self._model(dataset), normalization)


def _create_normalized_intensity(builder: IntensityBuilder, recipe: dict):
    model = builder._create_element(recipe["Intensity"])
    dataset = builder._kinematics.convert(builder._get_normalization_data())
    # its important to convert the dataset to tf tensors (memory footprint)
    dataset = {x: tf.constant(y) for x, y in dataset.items()}
    volume = atfi.const(builder._kinematics.phase_space_volume)
    return _NormalizedIntensity(model, dataset, volume)


class _StrengthIntensity:
    def __init__(self, intensity, strength):
        self._strength = strength
        self._intensity = intensity

    def __call__(self, dataset):
        return self._strength * self._intensity(dataset)


def _create_strength_intensity(builder, recipe):
    strength = builder._get_parameter(recipe["Strength"])
    intensity = builder._create_element(recipe["Intensity"])
    return _StrengthIntensity(intensity, strength)


class _IncoherentIntensity:
    def __init__(self, intensities):
        self._intensities = intensities

    def __call__(self, dataset):
        return tf.math.accumulate_n([y(dataset) for y in self._intensities])


def _create_incoherent_intensity(builder, recipe: dict):
    if not isinstance(recipe["Intensities"], list):
        raise Exception("Incoherent Intensity requires a list of intensities!")
    intensities = [builder._create_element(x) for x in recipe["Intensities"]]
    return _IncoherentIntensity(intensities)


class _CoherentIntensity:
    def __init__(self, amplitudes):
        self._amps = amplitudes

    def __call__(self, dataset):
        return tf.pow(
            tf.cast(
                tf.abs(tf.add_n([amp(dataset) for amp in self._amps])),
                dtype=tf.float64,
            ),
            tf.constant(2.0, dtype=tf.float64),
        )


def _create_coherent_intensity(builder, recipe: dict):
    if not isinstance(recipe["Amplitudes"], list):
        raise Exception("Coherent Intensity requires a list of amplitudes!")
    amps = [builder._create_element(x) for x in recipe["Amplitudes"]]
    return _CoherentIntensity(amps)


class _CoefficientAmplitude:
    def __init__(
        self, amplitude: Callable, mag: tf.Variable, phase: tf.Variable
    ):
        self._mag = mag
        self._phase = phase
        self._amp = amplitude

    def __call__(self, dataset):
        coefficient = atfi.complex(
            self._mag * atfi.cos(self._phase),
            self._mag * atfi.sin(self._phase),
        )
        return coefficient * self._amp(dataset)


def _create_coefficient_amplitude(builder, recipe: dict):
    if not isinstance(recipe, dict):
        raise ValueError("recipe corrupted!")

    pars = recipe["Parameters"]
    mag = builder._get_parameter(pars["Magnitude"])
    phase = builder._get_parameter(pars["Phase"])
    amp = builder._create_element(recipe["Amplitude"])
    return _CoefficientAmplitude(amp, mag, phase)


class _SequentialAmplitude:
    def __init__(self, amplitudes):
        self._seq_amps = amplitudes

    def __call__(self, dataset):
        seq_amp = atfi.complex(atfi.const(1.0), atfi.const(0.0))
        for amp in self._seq_amps:
            seq_amp = seq_amp * amp(dataset)
        return seq_amp


def _create_sequential_amplitude(builder, recipe: list):
    if not isinstance(recipe["Amplitudes"], list):
        raise Exception("Sequential Amplitude requires a list of amplitudes!")
    amp_recipes = recipe["Amplitudes"]
    if len(amp_recipes) == 0:
        raise Exception(
            "Sequential Amplitude requires a non-empty list of amplitudes!"
        )
    return _SequentialAmplitude(
        [builder._create_element(x) for x in amp_recipes]
    )


class _HelicityDecayParameters:
    def __init__(self):
        self.j = 0
        self.m = 0
        self.mprime = 0
        self.orbit_angular_momentum = 0
        self.theta_name = ""
        self.phi_name = ""
        self.resonance_mass = None
        self.resonance_width = None
        self.inv_mass_name = ""
        self.inv_mass_name_prod1 = ""
        self.inv_mass_name_prod2 = ""


class _HelicityDecay:
    def __init__(self, helicity_decay_params, dynamics_function):
        self._params = helicity_decay_params
        if dynamics_function:
            self._dynamics_function = dynamics_function
            self._call_wrapper = self._with_dynamics
        else:
            self._call_wrapper = self._without_dynamics

    def __call__(self, dataset):
        return self._call_wrapper(dataset)

    def _with_dynamics(self, dataset):
        return wigner_capital_d(
            dataset[self._params.phi_name],
            dataset[self._params.theta_name],
            0.0,
            2 * self._params.j,
            2 * self._params.m,
            2 * self._params.mprime,
        ) * self._dynamics_function(
            dataset[self._params.inv_mass_name],
            self._params.resonance_mass,
            self._params.resonance_width,
        )

    def _without_dynamics(self, dataset):
        return wigner_capital_d(
            dataset[self._params.phi_name],
            dataset[self._params.theta_name],
            0.0,
            2 * self._params.j,
            2 * self._params.m,
            2 * self._params.mprime,
        )


def _create_helicity_decay(builder, recipe: dict):
    if not isinstance(recipe, dict):
        raise Exception("Helicity Decay expects a dictionary recipe!")
    decaying_state = recipe["DecayParticle"]
    decay_products = recipe["DecayProducts"]["Particle"]

    def to_list(ids):
        return ids if isinstance(ids, list) else [ids]

    helicity_params = _HelicityDecayParameters()
    kinematics = builder._kinematics
    particle_list = builder._particles
    dynamics = builder._dynamics

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

    (
        helicity_params.inv_mass_name,
        helicity_params.theta_name,
        helicity_params.phi_name,
    ) = kinematics.register_subsystem(
        SubSystem(dec_prod_fs_ids, recoil_fs_ids, parent_recoil_fs_ids)
    )

    decaying_state_name = decaying_state["Name"]
    if decaying_state_name not in particle_list:
        raise LookupError(
            f"Could not find particle with name {decaying_state_name}"
        )
    particle_infos = particle_list[decaying_state_name]
    decay_dynamics = [x for x in dynamics if x["Name"] == decaying_state_name]

    if not decay_dynamics:
        raise LookupError(
            f"Could not find dynamics for particle with name"
            f" {decaying_state_name}"
        )
    decay_dynamics = decay_dynamics[0]

    non_dynamics_label = "nonDynamic"
    known_dynamics_functions = {
        "relativisticBreitWigner": relativistic_breit_wigner,
    }

    dynamics_type = decay_dynamics["Type"]
    dynamics_function = None
    if non_dynamics_label not in dynamics_type:
        if dynamics_type not in known_dynamics_functions:
            raise ValueError(
                f"Dynamics ({dynamics_type}) unknown. "
                f"Use one of the following: \n"
                f"{list(known_dynamics_functions.keys())}"
            )
        dynamics_function = known_dynamics_functions[dynamics_type]

    helicity_params.resonance_mass = builder._register_parameter(
        "Mass_" + decaying_state_name, particle_infos["Mass"]["Value"],
    )
    helicity_params.resonance_width = builder._register_parameter(
        "Width_" + decaying_state_name, particle_infos["Width"]["Value"],
    )

    # calculate spin based infos
    helicity_params.j = particle_infos["QuantumNumbers"]["Spin"]
    helicity_params.m = decaying_state["Helicity"]
    helicity_params.mprime = (
        decay_products[0]["Helicity"] - decay_products[1]["Helicity"]
    )

    helicity_params.orbit_angular_momentum = helicity_params.j

    # register decay product invariant masses
    helicity_params.inv_mass_name_prod1 = kinematics.register_invariant_mass(
        dec_prod_fs_ids[0]
    )
    helicity_params.inv_mass_name_prod2 = kinematics.register_invariant_mass(
        dec_prod_fs_ids[1]
    )

    return _HelicityDecay(helicity_params, dynamics_function)
