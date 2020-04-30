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
        phsp_data: dict = None,
    ):
        self._particles = particles
        self._dynamics: Dict[str, Any] = {}
        self._kinematics = kinematics
        self._parameters: Dict[str, tf.Variable] = {}
        self._phsp_data = phsp_data

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
            self._create_intensity(recipe["Intensity"]), self._parameters
        )

    def _create_intensity(self, recipe: dict) -> Callable:
        intensity_class = recipe["Class"]
        logging.debug("creating %s", intensity_class)

        # this dict could be used for builder injections later on
        basic_intensity_builders: Dict[str, Callable] = {
            "IncoherentIntensity": _IncoherentIntensity,
            "CoherentIntensity": _CoherentIntensity,
            "StrengthIntensity": _StrengthIntensity,
        }
        intensity_kinematics_builders = {
            "NormalizedIntensity": _NormalizedIntensity,
        }

        if intensity_class in basic_intensity_builders:
            return basic_intensity_builders[intensity_class](recipe, self)
        if intensity_class in intensity_kinematics_builders:
            return intensity_kinematics_builders[intensity_class](
                recipe, self, self._kinematics
            )

        raise Exception(f"Unknown intensity {intensity_class}!")

    def _create_amplitude(self, recipe: dict) -> Callable:
        amplitude_class = recipe["Class"]
        logging.debug("creating %s", amplitude_class)

        basic_amplitude_builders = {
            "CoefficientAmplitude": _CoefficientAmplitude,
            "SequentialAmplitude": _SequentialAmplitude,
        }
        general_amplitude_builders = {
            "HelicityDecay": _HelicityDecay,
        }

        if amplitude_class in basic_amplitude_builders:
            return basic_amplitude_builders[amplitude_class](recipe, self)
        if amplitude_class in general_amplitude_builders:
            return general_amplitude_builders[amplitude_class](
                recipe, self, self._kinematics, self._particles, self._dynamics
            )

        raise Exception(f"Unknown amplitude {amplitude_class}!")

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
        if not self._phsp_data:
            raise Exception(
                "No phase space sample given! This is required for the "
                "normalization."
            )
        return self._phsp_data


class _NormalizedIntensity:
    def __init__(self, recipe: dict, builder, kinematics):
        self._model_data = builder._create_intensity(recipe["Intensity"])
        self._model_phsp = builder._create_intensity(recipe["Intensity"])
        dataset = kinematics.convert(builder._get_normalization_data())
        # its important to convert the dataset to tf tensors (memory footprint)
        self._norm_dataset = {x: tf.constant(y) for x, y in dataset.items()}
        self._norm_volume = atfi.const(kinematics.phase_space_volume)

    @tf.function
    def __call__(self, dataset):
        normalization = tf.multiply(
            self._norm_volume,
            tf.reduce_mean(self._model_phsp(self._norm_dataset)),
        )
        return tf.divide(self._model_data(dataset), normalization)


class _StrengthIntensity:
    def __init__(self, recipe: dict, builder):
        if isinstance(recipe, dict):
            self.strength = builder._get_parameter(recipe["Strength"])
            self.intensity = builder._create_intensity(recipe["Intensity"])

    def __call__(self, dataset):
        return self.strength * self.intensity(dataset)


class _IncoherentIntensity:
    def __init__(self, recipe: dict, builder):
        if isinstance(recipe["Intensities"], list):
            intensity_recipes = list(recipe["Intensities"])
            self.intensities = [
                builder._create_intensity(x) for x in intensity_recipes
            ]
        else:
            raise Exception(
                "Incoherent Intensity requires a list of intensities!"
            )

    def __call__(self, dataset):
        return tf.math.accumulate_n([y(dataset) for y in self.intensities])


class _CoherentIntensity:
    def __init__(self, recipe: dict, builder):
        if isinstance(recipe["Amplitudes"], list):
            amp_recipes = recipe["Amplitudes"]
            self.amps = [builder._create_amplitude(x) for x in amp_recipes]
        else:
            raise Exception(
                "Coherent Intensity requires a list of intensities!"
            )

    def __call__(self, dataset):
        return tf.pow(
            tf.cast(
                tf.abs(tf.add_n([amp(dataset) for amp in self.amps])),
                dtype=tf.float64,
            ),
            tf.constant(2.0, dtype=tf.float64),
        )


class _CoefficientAmplitude:
    def __init__(self, recipe: dict, builder):
        if isinstance(recipe, dict):
            pars = recipe["Parameters"]
            self.mag = builder._get_parameter(pars["Magnitude"])
            self.phase = builder._get_parameter(pars["Phase"])

            self.amp = builder._create_amplitude(recipe["Amplitude"])

    def __call__(self, dataset):
        coefficient = atfi.complex(
            self.mag * atfi.cos(self.phase), self.mag * atfi.sin(self.phase)
        )
        return coefficient * self.amp(dataset)


class _SequentialAmplitude:
    def __init__(self, recipe: list, builder):
        if isinstance(recipe["Amplitudes"], list):
            amp_recipes = list(recipe["Amplitudes"])
            if len(amp_recipes) == 0:
                raise Exception(
                    "Sequential Amplitude requires a non-empty"
                    " list of amplitudes!"
                )
            self.seq_amp = [builder._create_amplitude(x) for x in amp_recipes]
        else:
            raise Exception(
                "Sequential Amplitude requires a list of amplitudes!"
            )

    def __call__(self, dataset):
        seq_amp = tf.complex(atfi.const(1.0), atfi.const(0.0))
        for amp in self.seq_amp:
            seq_amp = tf.multiply(seq_amp, amp(dataset))
        return seq_amp


class _HelicityDecay:
    def __init__(
        self, recipe: dict, builder, kinematics, particle_list, dynamics
    ):
        if isinstance(recipe, dict):
            decaying_state = recipe["DecayParticle"]
            decay_products = recipe["DecayProducts"]["Particle"]

            def to_list(ids):
                return ids if isinstance(ids, list) else [ids]

            # define the subsystem
            dec_prod_fs_ids = [x["FinalState"] for x in decay_products]
            dec_prod_fs_ids = [to_list(x) for x in dec_prod_fs_ids]

            recoil_fs_ids = []
            parent_recoil_fs_ids = []
            if "RecoilSystem" in recipe:
                if "RecoilFinalState" in recipe["RecoilSystem"]:
                    recoil_fs_ids = to_list(
                        recipe["RecoilSystem"]["RecoilFinalState"]
                    )
                if "ParentRecoilFinalState" in recipe["RecoilSystem"]:
                    parent_recoil_fs_ids = to_list(
                        recipe["RecoilSystem"]["ParentRecoilFinalState"]
                    )

            (
                self._inv_mass_name,
                self._theta_name,
                self._phi_name,
            ) = kinematics.register_subsystem(
                SubSystem(dec_prod_fs_ids, recoil_fs_ids, parent_recoil_fs_ids)
            )

            if decaying_state["Name"] not in particle_list:
                raise LookupError(
                    "Could not find particle with name "
                    + decaying_state["Name"]
                )
            particle_infos = particle_list[decaying_state["Name"]]
            decay_dynamics = [
                x for x in dynamics if x["Name"] == decaying_state["Name"]
            ]
            if not decay_dynamics:
                raise LookupError(
                    "Could not find dynamics for particle with "
                    "name " + decaying_state["Name"]
                )
            decay_dynamics = decay_dynamics[0]

            non_dynamics_label = "nonDynamic"
            known_dynamics_functions = {
                "relativisticBreitWigner": relativistic_breit_wigner,
            }

            dynamics_type = decay_dynamics["Type"]
            if non_dynamics_label in dynamics_type:
                self._call_wrapper = self._without_dynamics
            else:
                if dynamics_type not in known_dynamics_functions:
                    raise ValueError(
                        f"Dynamics ({dynamics_type}) unknown. "
                        f"Use one of the following: \n"
                        f"{list(known_dynamics_functions.keys())}"
                    )
                self._dynamics_function = known_dynamics_functions[
                    dynamics_type
                ]
                self._call_wrapper = self._with_dynamics

            self._resonance_mass = builder._register_parameter(
                "Mass_" + particle_infos["Name"],
                particle_infos["Mass"]["Value"],
            )
            self._resonance_width = builder._register_parameter(
                "Width_" + particle_infos["Name"],
                particle_infos["Width"]["Value"],
            )

            # calculate spin based infos
            self._j = particle_infos["QuantumNumbers"]["Spin"]
            self._m = decaying_state["Helicity"]
            self._mprime = (
                decay_products[0]["Helicity"] - decay_products[1]["Helicity"]
            )

            self._l = self._j

            # register decay product invariant masses
            self._inv_mass_name_prod1 = kinematics.register_invariant_mass(
                dec_prod_fs_ids[0]
            )
            self._inv_mass_name_prod2 = kinematics.register_invariant_mass(
                dec_prod_fs_ids[1]
            )
        else:
            raise Exception("Helicity Decay expects a dictionary recipe!")

    def __call__(self, dataset):
        self._call_wrapper(dataset)

    def _with_dynamics(self, dataset):
        return wigner_capital_d(
            dataset[self._phi_name],
            dataset[self._theta_name],
            0.0,
            2 * self._j,
            2 * self._m,
            2 * self._mprime,
        ) * self._dynamics_function(
            dataset[self._inv_mass_name],
            self._resonance_mass,
            self._resonance_width,
        )

    def _without_dynamics(self, dataset):
        return wigner_capital_d(
            dataset[self._phi_name],
            dataset[self._theta_name],
            0.0,
            2 * self._j,
            2 * self._m,
            2 * self._mprime,
        )
