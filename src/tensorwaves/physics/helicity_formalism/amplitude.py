"""Amplitude module for the helicity formalism.

Its responsibility is the construction of complicated helicity formalism
amplitude models using a recipe (see `IntensityBuilder`). These models are
encapsulated in an `IntensityTF` class, which can be evaluated as a regular
callable.
"""

import logging
from typing import (
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
)

import amplitf.interface as atfi
import expertsystem.amplitude.model as es
import numpy as np
import tensorflow as tf
from amplitf.dynamics import (
    blatt_weisskopf_ff_squared,
    relativistic_breit_wigner,
)
from amplitf.kinematics import two_body_momentum_squared, wigner_capital_d
from expertsystem.particle import Particle, ParticleCollection
from sympy.physics.quantum.cg import CG

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

    def __call__(self, dataset: Dict[str, np.ndarray]) -> np.ndarray:
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
    def parameters(self) -> Dict[str, tf.Variable]:
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
        particles: ParticleCollection,
        kinematics: HelicityKinematics,
        phsp_data: Optional[np.ndarray] = None,
    ):
        self._particles = particles
        self._dynamics: Optional[es.ParticleDynamics] = None
        self._kinematics = kinematics
        self._parameters: Dict[str, tf.Variable] = {}
        if phsp_data is None:
            phsp_data = np.array([])
        self._phsp_data = phsp_data
        self._registered_element_builders: Dict[Type[es.Node], Callable] = {
            es.NormalizedIntensity: _create_normalized_intensity,
            es.StrengthIntensity: _create_strength_intensity,
            es.IncoherentIntensity: _create_incoherent_intensity,
            es.CoherentIntensity: _create_coherent_intensity,
            es.CoefficientAmplitude: _create_coefficient_amplitude,
            es.SequentialAmplitude: _create_sequential_amplitude,
            es.HelicityDecay: _create_helicity_decay,
            es.CanonicalDecay: _create_helicity_decay,
        }
        self._registered_dynamics_builders: Dict[
            Type[es.Dynamics], Callable
        ] = {
            es.NonDynamic: _NonDynamic,
            es.RelativisticBreitWigner: _RelativisticBreitWigner,
        }

    def create_intensity(self, model: es.AmplitudeModel) -> IntensityTF:
        """Create an `IntensityTF` instance based on a recipe.

        Args:
            model: Contains builder instructions. These recipe files can be
                generated via the expert system (see
                :doc:`expertsystem:usage/workflow`).
        """
        self._dynamics = model.dynamics
        self._initialize_parameters(model)
        return IntensityTF(
            self.create_element(model.intensity), self._parameters
        )

    def create_element(self, intensity_node: es.Node) -> Callable:
        """Create a computation element from the recipe.

        The recipe can only contain names registered in the pool of known
        element builders.
        """
        element_class = type(intensity_node)
        logging.debug("creating %s", element_class)

        if element_class not in self._registered_element_builders:
            raise Exception(f"Unknown element {element_class.__name__}!")

        return self._registered_element_builders[element_class](
            self, intensity_node, kinematics=self._kinematics
        )

    def create_dynamics(
        self,
        decaying_state: Particle,
        dynamics_properties: "DynamicsProperties",
    ) -> Callable:
        """Create a dynamics function callable."""
        if self._dynamics is None:
            raise ValueError("Dynamics has not yet been set")

        if decaying_state.name not in self._dynamics:
            self._dynamics.set_breit_wigner(decaying_state.name)
        decay_dynamics = self._dynamics[decaying_state.name]

        kwargs = {}
        form_factor = getattr(decay_dynamics, "form_factor", None)
        if isinstance(form_factor, es.BlattWeisskopf):
            kwargs.update({"form_factor": "BlattWeisskopf"})
            meson_radius_val = form_factor.meson_radius.value
            meson_radius_par = self.register_parameter(
                f"MesonRadius_{decaying_state.name}",
                meson_radius_val,
            )
            dynamics_properties = dynamics_properties._replace(
                meson_radius=meson_radius_par
            )

        dynamics_builder = self._get_dynamics_builder(decaying_state.name)
        return dynamics_builder(dynamics_properties, **kwargs)

    def register_dynamics_builder(
        self,
        dynamics_name: Type[es.Dynamics],
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
    ) -> Callable[..., Callable]:
        if self._dynamics is None:
            raise ValueError("Dynamics has not been set")

        dynamics = self._dynamics[decaying_state_name]
        if type(dynamics) not in self._registered_dynamics_builders:
            raise ValueError(
                f"Dynamics ({dynamics.__class__.__name__}) unknown. "
                f"Use one of the following: \n"
                f"{list(self._registered_dynamics_builders.keys())}"
            )
        return self._registered_dynamics_builders[type(dynamics)]

    def register_parameter(self, name: str, value: float) -> tf.Variable:
        if name not in self._parameters:
            self._parameters[name] = tf.Variable(
                value, name=name, dtype=tf.float64
            )
        return self._parameters[name]

    def get_parameter(self, name: str) -> tf.Variable:
        if name not in self._parameters:
            raise Exception(f'Parameter "{name}" not registered')

        return self._parameters[name]

    def _initialize_parameters(self, model: es.AmplitudeModel) -> None:
        parameters: List[es.FitParameter] = list(model.parameters.values())
        for par in parameters:
            self._parameters[par.name] = tf.Variable(
                par.value, name=par.name, dtype=tf.float64
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
            self._norm_volume,
            tf.reduce_mean(self._model(self._norm_dataset)),
        )
        return tf.divide(self._model(dataset), normalization)


def _create_normalized_intensity(
    builder: IntensityBuilder, node: es.Node, **_: dict
) -> Callable:
    if not isinstance(node, es.NormalizedIntensity):
        raise TypeError(
            f"Requires {es.NormalizedIntensity.__class__.__name__}"
        )
    model = builder.create_element(node.intensity)
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
    builder: IntensityBuilder, node: es.Node, **_: dict
) -> Callable:
    if not isinstance(node, es.StrengthIntensity):
        raise TypeError
    strength = builder.get_parameter(node.strength.name)
    intensity = builder.create_element(node.intensity)
    return _StrengthIntensity(intensity, strength)


class _IncoherentIntensity:
    def __init__(self, intensities: List[Callable]) -> None:
        self._intensities = intensities

    def __call__(self, dataset: dict) -> tf.Tensor:
        return tf.math.accumulate_n([y(dataset) for y in self._intensities])


def _create_incoherent_intensity(
    builder: IntensityBuilder, node: es.Node, **_: dict
) -> Callable:
    if not isinstance(node, es.IncoherentIntensity):
        raise TypeError
    intensities = [builder.create_element(x) for x in node.intensities]
    return _IncoherentIntensity(intensities)


class _CoherentIntensity:
    def __init__(self, amplitudes: List[Callable]) -> None:
        self._amps = amplitudes

    def __call__(self, dataset: dict) -> tf.Tensor:
        return tf.pow(
            tf.cast(  # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
                tf.abs(tf.add_n([amp(dataset) for amp in self._amps])),
                dtype=tf.float64,
            ),
            tf.constant(2.0, dtype=tf.float64),
        )


def _create_coherent_intensity(
    builder: IntensityBuilder, node: es.Node, **_: dict
) -> Callable:
    if not isinstance(node, es.CoherentIntensity):
        raise TypeError
    amplitudes = [builder.create_element(x) for x in node.amplitudes]
    return _CoherentIntensity(amplitudes)


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
    builder: IntensityBuilder, node: es.Node, **_: dict
) -> Callable:
    if not isinstance(node, es.CoefficientAmplitude):
        raise TypeError
    magnitude = builder.get_parameter(node.magnitude.name)
    phase = builder.get_parameter(node.phase.name)
    amplitude = builder.create_element(node.amplitude)
    return _CoefficientAmplitude(amplitude, magnitude, phase)


class _SequentialAmplitude:
    def __init__(self, amplitudes: List[Callable]) -> None:
        self._seq_amps = amplitudes

    def __call__(self, dataset: dict) -> tf.Tensor:
        seq_amp = atfi.complex(atfi.const(1.0), atfi.const(0.0))
        for amp in self._seq_amps:
            seq_amp = seq_amp * amp(dataset)
        return seq_amp


def _create_sequential_amplitude(
    builder: IntensityBuilder, node: es.Node, **_: dict
) -> Callable:
    if not isinstance(node, es.SequentialAmplitude):
        raise TypeError
    if len(node.amplitudes) == 0:
        raise Exception(
            "Sequential Amplitude requires a non-empty list of amplitudes!"
        )
    return _SequentialAmplitude(
        [builder.create_element(x) for x in node.amplitudes]
    )


class _AngularProperties(NamedTuple):
    j: float
    m: float
    mprime: float
    theta_name: str
    phi_name: str


class DynamicsProperties(NamedTuple):
    """Data structure representing dynamic properties."""

    orbit_angular_momentum: float
    resonance_mass: float
    resonance_width: float
    inv_mass_name: str
    inv_mass_name_prod1: str
    inv_mass_name_prod2: str
    meson_radius: Optional[float]


class _RelativisticBreitWigner:
    def __init__(
        self,
        dynamics_props: DynamicsProperties,
        form_factor: Optional[str] = None,
    ) -> None:
        self._dynamics_props = dynamics_props
        self._call_wrapper = self._without_form_factor
        if form_factor == "BlattWeisskopf":
            self._call_wrapper = self._with_form_factor

    def __call__(self, dataset: dict) -> tf.Tensor:
        return self._call_wrapper(dataset)

    def _without_form_factor(self, dataset: dict) -> tf.Tensor:
        return relativistic_breit_wigner(
            dataset[self._dynamics_props.inv_mass_name],
            self._dynamics_props.resonance_mass,
            self._dynamics_props.resonance_width,
        )

    def _with_form_factor(self, dataset: dict) -> tf.Tensor:
        inv_mass_squared = dataset[self._dynamics_props.inv_mass_name]
        inv_mass = atfi.sqrt(inv_mass_squared)
        mass0 = self._dynamics_props.resonance_mass
        gamma0 = self._dynamics_props.resonance_width
        m_a = atfi.sqrt(dataset[self._dynamics_props.inv_mass_name_prod1])
        m_b = atfi.sqrt(dataset[self._dynamics_props.inv_mass_name_prod2])
        meson_radius = self._dynamics_props.meson_radius
        l_orbit = self._dynamics_props.orbit_angular_momentum
        q_squared = two_body_momentum_squared(inv_mass, m_a, m_b)
        q0_squared = two_body_momentum_squared(mass0, m_a, m_b)
        ff2 = blatt_weisskopf_ff_squared(q_squared, meson_radius, l_orbit)
        ff02 = blatt_weisskopf_ff_squared(q0_squared, meson_radius, l_orbit)
        width = gamma0 * (mass0 / inv_mass) * (ff2 / ff02)
        # So far its all in float64,
        # but for the sqrt operation it has to be converted to complex
        width = atfi.complex(
            width, tf.constant(0.0, dtype=tf.float64)
        ) * atfi.sqrt(
            atfi.complex(
                (q_squared / q0_squared),
                tf.constant(0.0, dtype=tf.float64),
            )
        )
        return relativistic_breit_wigner(
            inv_mass_squared, mass0, width
        ) * atfi.complex(mass0 * gamma0 * atfi.sqrt(ff2), atfi.const(0.0))


class _NonDynamic:
    def __init__(
        self,
        dynamics_props: DynamicsProperties,
        form_factor: Optional[es.FormFactor] = None,
    ) -> None:
        self._dynamics_props = dynamics_props
        self._call_wrapper: Callable[
            [dict], tf.Tensor
        ] = self._without_form_factor
        if isinstance(form_factor, es.BlattWeisskopf):
            self._call_wrapper = self._with_form_factor

    def __call__(self, dataset: dict) -> tf.Tensor:
        return self._call_wrapper(dataset)

    @staticmethod
    def _without_form_factor(_: dict) -> tf.Tensor:
        return tf.complex(
            tf.constant(1.0, dtype=tf.float64),
            tf.constant(0.0, dtype=tf.float64),
        )

    def _with_form_factor(self, dataset: dict) -> tf.Tensor:
        inv_mass = atfi.sqrt(dataset[self._dynamics_props.inv_mass_name])
        m_a = atfi.sqrt(dataset[self._dynamics_props.inv_mass_name_prod1])
        m_b = atfi.sqrt(dataset[self._dynamics_props.inv_mass_name_prod2])
        meson_radius = self._dynamics_props.meson_radius
        l_orbit = self._dynamics_props.orbit_angular_momentum

        q_squared = two_body_momentum_squared(inv_mass, m_a, m_b)

        return atfi.complex(
            atfi.sqrt(
                blatt_weisskopf_ff_squared(q_squared, meson_radius, l_orbit)
            ),
            atfi.const(0.0),
        )


class _HelicityDecay:
    def __init__(
        self,
        angular_params: "_AngularProperties",
        dynamics_function: Callable,
        prefactor: float = 1.0,
    ) -> None:
        self._params = angular_params
        self._dynamics_function = dynamics_function
        self._prefactor = prefactor

    def __call__(self, dataset: dict) -> tf.Tensor:
        return (
            self._prefactor
            * wigner_capital_d(
                dataset[self._params.phi_name],
                dataset[self._params.theta_name],
                0.0,
                int(2 * self._params.j),
                int(2 * self._params.m),
                int(2 * self._params.mprime),
            )
            * self._dynamics_function(dataset)
        )


def _clebsch_gordan_coefficient(clebsch_gordan: es.ClebschGordan) -> float:
    return (
        CG(
            j1=clebsch_gordan.j_1,
            m1=clebsch_gordan.m_1,
            j2=clebsch_gordan.j_2,
            m2=clebsch_gordan.m_2,
            j3=clebsch_gordan.J,
            m3=clebsch_gordan.M,
        )
        .doit()
        .evalf()
    )


def _determine_canonical_prefactor(node: es.CanonicalDecay) -> float:
    l_s = _clebsch_gordan_coefficient(node.l_s)
    s2s3 = _clebsch_gordan_coefficient(node.s2s3)
    return l_s * s2s3


def _create_helicity_decay(  # pylint: disable=too-many-locals
    builder: IntensityBuilder,
    node: es.Node,
    kinematics: HelicityKinematics,
) -> Callable:
    if not isinstance(node, (es.HelicityDecay, es.CanonicalDecay)):
        raise TypeError
    decaying_state = node.decaying_particle
    decay_products = node.decay_products
    dec_prod_fs_ids = [x.final_state_ids for x in decay_products]

    recoil_final_state = []
    parent_recoil_final_state = []
    recoil_system = node.recoil_system
    if recoil_system is not None:
        recoil_final_state = recoil_system.recoil_final_state
        if recoil_system.parent_recoil_final_state is not None:
            parent_recoil_final_state = recoil_system.parent_recoil_final_state

    inv_mass_name, theta_name, phi_name = kinematics.register_subsystem(
        SubSystem(
            dec_prod_fs_ids,
            recoil_final_state,
            parent_recoil_final_state,
        )
    )

    particle = decaying_state.particle
    j = particle.spin

    prefactor = 1.0
    if isinstance(node, es.CanonicalDecay):
        prefactor = _determine_canonical_prefactor(node)

    dynamics = _create_dynamics(
        builder,
        node,
        dec_prod_fs_ids,
        decaying_state,
        inv_mass_name,
        kinematics,
    )

    return _HelicityDecay(
        _AngularProperties(
            j=j,
            m=decaying_state.helicity,
            mprime=decay_products[0].helicity - decay_products[1].helicity,
            theta_name=theta_name,
            phi_name=phi_name,
        ),
        dynamics,
        prefactor=prefactor,
    )


def _create_dynamics(
    builder: IntensityBuilder,
    amplitude_node: es.AmplitudeNode,
    dec_prod_fs_ids: Sequence,
    decaying_state: es.HelicityParticle,
    inv_mass_name: str,
    kinematics: HelicityKinematics,
) -> Callable:
    particle = decaying_state.particle
    orbit_angular_momentum = particle.spin
    if isinstance(amplitude_node, es.CanonicalDecay):
        orbit_angular_momentum = amplitude_node.l_s.j_1
        if not orbit_angular_momentum.is_integer():
            raise ValueError(
                "Model invalid! Using a non integer value for the angular"
                " orbital momentum L. Seems like you are using the helicity"
                " formalism, but should be using the canonical formalism"
            )

    dynamics = builder.create_dynamics(
        particle,
        DynamicsProperties(
            orbit_angular_momentum=orbit_angular_momentum,
            resonance_mass=builder.register_parameter(
                f"Mass_{particle.name}",
                particle.mass,
            ),
            resonance_width=builder.register_parameter(
                f"Width_{particle.name}",
                particle.width,
            ),
            inv_mass_name=inv_mass_name,
            inv_mass_name_prod1=kinematics.register_invariant_mass(
                dec_prod_fs_ids[0]
            ),
            inv_mass_name_prod2=kinematics.register_invariant_mass(
                dec_prod_fs_ids[1]
            ),
            meson_radius=None,
        ),
    )
    return dynamics
