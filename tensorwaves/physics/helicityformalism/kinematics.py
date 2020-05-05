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
from typing import Union, TypeVar

import amplitf.kinematics as tfa_kin

import numpy as np

from tensorwaves.interfaces import Kinematics

from tensorwaves.physics.particle import extract_particles


class SubSystem:
    """Represents a part of a decay chain.

    A SubSystem resembles a decaying state and its ingoing and outgoing state.
    It is uniquely defined by

    * :attr:`final_states`
    * :attr:`recoil_state`
    * :attr:`parent_recoil_state`
    """

    def __init__(self, final_states, recoil_state, parent_recoil_state):
        """Fully initializes the :class:`~SubSystem`.

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
    def final_states(self):
        """Get final state content of the decay products."""
        return self._final_states

    @property
    def recoil_state(self):
        """Get final state content of the recoil partner."""
        return self._recoil_state

    @property
    def parent_recoil_state(self):
        """Get final state content of the recoil partner of the parent."""
        return self._parent_recoil_state

    def __eq__(self, other):
        """Equal testing operator."""
        if self._final_states != other._final_states:
            return False
        if self._recoil_state != other._recoil_state:
            return False
        if self._parent_recoil_state != other._parent_recoil_state:
            return False
        return True

    def __hash__(self):
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

    T = TypeVar("HelicityKinematics", bound="HelicityKinematics")

    def __init__(
        self,
        initial_state_mass,
        final_state_masses,
        fs_id_event_pos_mapping=None,
    ):
        """Initialize the a blank HelicityKinematics.

        Args:
            fs_id_event_pos_mapping: Optional mapping between particle unique
                ids and the position in the event array.
        """
        self._initial_state_mass = initial_state_mass
        self._final_state_masses = final_state_masses
        self._registered_inv_masses = dict()
        self._registered_subsystems = dict()
        self._fs_id_event_pos_mapping = fs_id_event_pos_mapping

    @classmethod
    def from_recipe(cls, recipe: dict) -> T:
        particles = extract_particles(recipe)
        fi_state = recipe["Kinematics"]["FinalState"]
        in_state = recipe["Kinematics"]["InitialState"]
        return cls(
            particles[in_state[0]["Particle"]]["Mass"]["Value"],
            [particles[x["Particle"]]["Mass"]["Value"] for x in fi_state],
            {x["ID"]: pos for pos, x in enumerate(fi_state)},
        )

    @property
    def initial_state_mass(self) -> float:
        return self._initial_state_mass

    @property
    def final_state_masses(self) -> list:
        return self._final_state_masses

    @property
    def phase_space_volume(self):
        """Get volume of the defined phase space.

        Return:
            `float`
        """
        return 1.0

    def is_within_phase_space(self, events):
        """Check whether events lie within the phase space definition."""
        raise NotImplementedError("Currently not implemented.")

    def register_invariant_mass(self, final_state: Union[tuple, list]):
        """Register an invariant mass :math:`s`.

        Args:
            final_state: collection of particle unique id's

        Return:
            A `str` key representing the invariant mass. It can be used to
            retrieve this invariant mass from the dataset returned by
            :meth:`~convert`.
        """
        logging.debug("registering inv mass in kinematics")
        final_state = tuple(final_state)
        if final_state not in self._registered_inv_masses:
            label = "mSq"
            for particle_uid in final_state:
                label += "_" + str(particle_uid)

            self._registered_inv_masses[final_state] = label
        return self._registered_inv_masses[final_state]

    def register_helicity_angles(self, subsystem: SubSystem):
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

    def register_subsystem(self, subsystem: SubSystem):
        r"""Register all kinematic variables of the :class:`~SubSystem`.

        Args:
            subsystem: SubSystem to which the registered kinematic variables
                correspond.

        Return:
            A tuple of `str` keys representing the :math:`(s, \theta, \phi)`.
            They can be used to retrieve the kinematic data from the dataset
            returned by :meth:`~convert`.
        """
        state_fs = []
        for fs_uid in subsystem.final_states:
            state_fs += fs_uid
        invmass_name = self.register_invariant_mass(list(set(state_fs)))
        angle_names = self.register_helicity_angles(subsystem)

        return (invmass_name,) + angle_names

    def _convert_ids_to_indices(self, ids: tuple):
        """Convert unique ids to event indices.

        Uses the :attr:`_fs_id_event_pos_mapping`.
        """
        if self._fs_id_event_pos_mapping:
            return [self._fs_id_event_pos_mapping[i] for i in ids]

        return ids

    def convert(self, events):
        r"""Convert events to the registered kinematics variables.

        Args:
            events: A three dimensional numpy array of the shape
                :math:`(n_{\mathrm{part}}, n_{\mathrm{evts}}, 4)`.

                * :math:`n_{\mathrm{part}}` is the number of particles
                * :math:`n_{\mathrm{evts}}` is the number of events

                The third dimension correspond to the four momentum info
                :math:`(p_x, p_y, p_z, E)`.

        Return:
            A `dict` containing the registered kinematic variables as keys
            and their corresponding values. This is also known as a dataset.
        """
        logging.info("converting %s events", len(events[0]))

        dataset = {}

        for (
            four_momenta_ids,
            inv_mass_name,
        ) in self._registered_inv_masses.items():
            four_momenta = np.sum(
                events[self._convert_ids_to_indices(four_momenta_ids), :],
                axis=0,
            )

            dataset[inv_mass_name] = tfa_kin.mass_squared(
                np.array(four_momenta)
            )

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
