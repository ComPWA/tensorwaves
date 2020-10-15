r"""Kinematic based calculations for the helicity formalism.

It's responsibilities are defined by the interface
:class:`.interfaces.Kinematics`.

Here, the main responsibility is the conversion of general kinematic
information of a reaction to helicity formalism specific quantities

:math:`(s, \theta, \phi)`

The basic building blocks are the :class:`~HelicityKinematics` and
:class:`~SubSystem`.
"""
import logging
from collections import abc
from typing import Dict, List, Optional, Sequence, Tuple

import amplitf.kinematics as tfa_kin
import numpy as np
from expertsystem.amplitude.model import AmplitudeModel
from expertsystem.particle import ParticleCollection

from tensorwaves.interfaces import Kinematics


class ParticleReactionKinematicsInfo:
    r"""Contains boundary condition information of a particle reaction.

    Args:
        initial_state_names: Defines the initial state
        final_state_names: Defines the final state
        particle_dict: Contains particle information
        total_invariant_mass: Invariant mass :math:`\sqrt(s)` of the initial or
            final state. Has to be specified for a multi particle initial state.

        fs_id_event_pos_mapping: Mapping between particle IDs and their
            positions in an event collection.

    """

    def __init__(
        self,
        initial_state_names: List[str],
        final_state_names: List[str],
        particles: ParticleCollection,
        total_invariant_mass: Optional[float] = None,
        fs_id_event_pos_mapping: Optional[Dict[int, int]] = None,
    ):
        if isinstance(initial_state_names, str):
            initial_state_names = (initial_state_names,)
        if len(initial_state_names) == 0:
            raise ValueError("initial_state_names cannot be empty!")
        if len(final_state_names) == 0:
            raise ValueError("final_state_names cannot be empty!")

        self._initial_state_particles = [
            particles[x] for x in initial_state_names
        ]
        self._final_state_particles = [particles[x] for x in final_state_names]

        if len(self._initial_state_particles) == 1:
            if total_invariant_mass:
                logging.warning(
                    "Total invariant mass sqrt(s) given with a single particle"
                    " initial state! Using given sqrt(s)!"
                )
            else:
                mass = self._initial_state_particles[0].mass
                self._total_invariant_mass = mass
        else:
            if not total_invariant_mass:
                raise ValueError("Total invariant mass sqrt(s) not given!")
            self._total_invariant_mass = total_invariant_mass

        self._fs_id_event_pos_mapping = fs_id_event_pos_mapping

    @classmethod
    def from_model(
        cls, model: AmplitudeModel
    ) -> "ParticleReactionKinematicsInfo":
        """Initialize from a recipe dictionary."""
        particles = model.particles
        fi_state = model.kinematics.final_state
        in_state = model.kinematics.initial_state
        fs_id_event_pos_mapping = {
            state_id: pos for pos, state_id in enumerate(fi_state)
        }
        return cls(
            initial_state_names=[p.name for p in in_state.values()],
            final_state_names=[p.name for p in fi_state.values()],
            particles=particles,
            fs_id_event_pos_mapping=fs_id_event_pos_mapping,
        )

    @property
    def initial_state_masses(self) -> List[float]:
        return [p.mass for p in self._initial_state_particles]

    @property
    def final_state_masses(self) -> List[float]:
        return [p.mass for p in self._final_state_particles]

    @property
    def total_invariant_mass(self) -> float:
        return self._total_invariant_mass

    @property
    def fs_id_event_pos_mapping(self) -> Optional[Dict[int, int]]:
        return self._fs_id_event_pos_mapping


class SubSystem(abc.Hashable):
    """Represents a part of a decay chain.

    A SubSystem resembles a decaying state and its ingoing and outgoing state.
    It is uniquely defined by:

    * :attr:`final_states`
    * :attr:`recoil_state`
    * :attr:`parent_recoil_state`
    """

    def __init__(
        self,
        final_states: Sequence[Sequence[int]],
        recoil_state: Sequence[int],
        parent_recoil_state: Sequence[int],
    ) -> None:
        """Fully initialize the :class:`SubSystem`.

        Args:
            final_states: `tuple` of `tuple` s containing unique ids.
                Represents the final state content of the decay products.
            recoil_state: `tuple` of unique ids representing the recoil partner
                of the decaying state.
            parent_recoil_state: `tuple` of unique ids representing the recoil
                partner of the parent state.

        """
        self._final_states = tuple(tuple(x) for x in final_states)
        self._recoil_state = tuple(recoil_state)
        self._parent_recoil_state = tuple(parent_recoil_state)

    @property
    def final_states(self) -> Tuple[tuple, ...]:
        """Get final state content of the decay products."""
        return self._final_states

    @property
    def recoil_state(self) -> tuple:
        """Get final state content of the recoil partner."""
        return self._recoil_state

    @property
    def parent_recoil_state(self) -> tuple:
        """Get final state content of the recoil partner of the parent."""
        return self._parent_recoil_state

    def __eq__(self, other: object) -> bool:
        """Equal testing operator."""
        if not isinstance(other, SubSystem):
            raise NotImplementedError
        if self._final_states != other._final_states:
            return False
        if self._recoil_state != other._recoil_state:
            return False
        if self._parent_recoil_state != other._parent_recoil_state:
            return False
        return True

    def __hash__(self) -> int:
        """Hash function to use SubSystem as key."""
        return hash(
            (self._final_states, self._recoil_state, self._parent_recoil_state)
        )


class HelicityKinematics(Kinematics):
    """Kinematics of the helicity formalism.

    General usage is

        1. Register kinematic variables via the three methods
           (:meth:`register_invariant_mass`, :meth:`register_helicity_angles`,
           :meth:`register_subsystem`) first.
        2. Then convert events to these kinematic variables.

    For additional functionality check :meth:`phase_space_volume` and
    :meth:`is_within_phase_space`.
    """

    def __init__(self, reaction_info: ParticleReactionKinematicsInfo):
        """Initialize the a blank HelicityKinematics.

        Args:
            reaction_info: data structure that contains all of the kinematic
                information of the particle reaction.

        """
        self._reaction_info = reaction_info
        self._registered_inv_masses: Dict[Tuple, str] = dict()
        self._registered_subsystems: Dict[SubSystem, Tuple[str, str]] = dict()

    @classmethod
    def from_model(cls, model: AmplitudeModel) -> "HelicityKinematics":
        return cls(ParticleReactionKinematicsInfo.from_model(model))

    @property
    def reaction_kinematics_info(self) -> ParticleReactionKinematicsInfo:
        return self._reaction_info

    @property
    def phase_space_volume(self) -> float:
        return 1.0

    def is_within_phase_space(self, events: np.ndarray) -> Tuple[bool]:
        """Check whether events lie within the phase space definition."""
        raise NotImplementedError

    def register_invariant_mass(self, final_state: Sequence) -> str:
        """Register an invariant mass :math:`s`.

        Args:
            final_state: collection of particle unique id's

        Return:
            A `str` key representing the invariant mass. It can be used to
            retrieve this invariant mass from the dataset returned by
            :meth:`~convert`.

        """
        logging.debug("registering inv mass in kinematics")
        _final_state: tuple = tuple(sorted(final_state))
        if _final_state not in self._registered_inv_masses:
            label = "mSq"
            for particle_uid in _final_state:
                label += "_" + str(particle_uid)

            self._registered_inv_masses[_final_state] = label
        return self._registered_inv_masses[_final_state]

    def register_helicity_angles(
        self, subsystem: SubSystem
    ) -> Tuple[str, str]:
        r"""Register helicity angles :math:`(\theta, \phi)` of a `SubSystem`.

        Args:
            subsystem: SubSystem to which the registered angles correspond.

        Return:
            A pair of `str` keys representing the angles. They can be used to
            retrieve the angles from the dataset returned by :meth:`~convert`.

        """
        logging.debug("registering helicity angles in kinematics")
        if subsystem not in self._registered_subsystems:
            suffix = ""
            for final_state in subsystem.final_states:
                suffix += "+"
                for particle_uid in final_state:
                    suffix += str(particle_uid) + "_"
                suffix = suffix[:-1]
            if subsystem.recoil_state:
                suffix += "_vs_"
                for particle_uid in subsystem.recoil_state:
                    suffix += str(particle_uid) + "_"
                suffix = suffix[:-1]

            self._registered_subsystems[subsystem] = (
                "theta" + suffix,
                "phi" + suffix,
            )
        return self._registered_subsystems[subsystem]

    def register_subsystem(self, subsystem: SubSystem) -> Tuple[str, ...]:
        r"""Register all kinematic variables of the :class:`~SubSystem`.

        Args:
            subsystem: SubSystem to which the registered kinematic variables
                correspond.

        Return:
            A tuple of `str` keys representing the :math:`(s, \theta, \phi)`.
            They can be used to retrieve the kinematic data from the dataset
            returned by :meth:`~convert`.

        """
        state_fs: list = []
        for fs_uid in subsystem.final_states:
            state_fs += fs_uid
        invmass_name = self.register_invariant_mass(list(set(state_fs)))
        angle_names = self.register_helicity_angles(subsystem)

        return (invmass_name,) + angle_names

    def _convert_ids_to_indices(self, ids: Tuple[int, ...]) -> Tuple[int, ...]:
        if self._reaction_info.fs_id_event_pos_mapping:
            return tuple(
                self._reaction_info.fs_id_event_pos_mapping[i] for i in ids
            )
        return ids

    def convert(self, events: np.ndarray) -> dict:
        r"""Convert events to the registered kinematics variables.

        Args:
            events: A three dimensional numpy array of the shape
                :math:`(n_{\mathrm{part}}, n_{\mathrm{events}}, 4)`.

                * :math:`n_{\mathrm{part}}` is the number of particles
                * :math:`n_{\mathrm{events}}` is the number of events

                The third dimension correspond to the four momentum info
                :math:`(p_x, p_y, p_z, E)`.

        Return:
            A `dict` containing the registered kinematic variables as keys
            and their corresponding values. This is also known as a dataset.

        """
        logging.debug("converting %s events", len(events[0]))

        dataset = {}

        for (
            four_momenta_ids,
            inv_mass_name,
        ) in self._registered_inv_masses.items():
            if len(four_momenta_ids) == 1:
                index = self._convert_ids_to_indices(four_momenta_ids)[0]

                dataset[inv_mass_name] = np.square(
                    np.array(self._reaction_info.final_state_masses[index])
                )

            else:
                four_momenta = np.sum(
                    events[self._convert_ids_to_indices(four_momenta_ids), :],
                    axis=0,
                )

                dataset[inv_mass_name] = tfa_kin.mass_squared(
                    np.array(four_momenta)
                ).numpy()

        for subsys, angle_names in self._registered_subsystems.items():
            topology = [
                np.sum(events[self._convert_ids_to_indices(x), :], axis=0)
                for x in subsys.final_states
            ]
            if subsys.recoil_state:
                topology = [
                    topology,
                    np.sum(
                        events[
                            self._convert_ids_to_indices(subsys.recoil_state),
                            :,
                        ],
                        axis=0,
                    ),
                ]
            if subsys.parent_recoil_state:
                topology = [
                    topology,
                    np.sum(
                        events[
                            self._convert_ids_to_indices(
                                subsys.parent_recoil_state
                            ),
                            :,
                        ],
                        axis=0,
                    ),
                ]

            values = tfa_kin.nested_helicity_angles(topology)

            # the last two angles is always what we are interested
            dataset[angle_names[0]] = values[-2].numpy()
            dataset[angle_names[1]] = values[-1].numpy()

        return dataset
