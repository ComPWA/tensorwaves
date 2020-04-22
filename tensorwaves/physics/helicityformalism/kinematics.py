import amplitf.kinematics as tfa_kin
import logging
import numpy as np
from tensorwaves.interfaces import Kinematics


class SubSystem():
    def __init__(self, final_states, recoil_state, parent_recoil_state):
        self._final_states = tuple(tuple(x) for x in final_states)
        self._recoil_state = tuple(recoil_state)
        self._parent_recoil_state = tuple(parent_recoil_state)

    @property
    def final_states(self):
        return self._final_states

    @property
    def recoil_state(self):
        return self._recoil_state

    @property
    def parent_recoil_state(self):
        return self._parent_recoil_state

    def __eq__(self, other):
        if self._final_states != other._final_states:
            return False
        if self._recoil_state != other._recoil_state:
            return False
        if self._parent_recoil_state != other._parent_recoil_state:
            return False
        return True

    def __hash__(self):
        return hash(
            (self._final_states,
             self._recoil_state,
             self._parent_recoil_state)
        )


def _add_four_vectors(four_vectors):
    if len(four_vectors) == 0:
        return four_vectors
    return np.sum(four_vectors, axis=0)


class HelicityKinematics(Kinematics):
    def __init__(self, fs_id_event_pos_mapping=None):
        self._registered_inv_masses = dict()
        self._registered_subsystems = dict()
        self._fs_id_event_pos_mapping = fs_id_event_pos_mapping

    @property
    def phase_space_volume(self):
        return 1.0  # TODO: calculate phsp volume based on final state

    def reduce_to_phase_space(self, events):
        return events  # TODO: remove events which are outside domain

    def register_invariant_mass(self, final_state: tuple or list):
        logging.debug("registering inv mass in kinematics")
        final_state = tuple(final_state)
        if final_state not in self._registered_inv_masses:
            label = 'mSq'
            for x in final_state:
                label += '_' + str(x)

            self._registered_inv_masses[final_state] = label
        return self._registered_inv_masses[final_state]

    def register_helicity_angles(self, subsystem: SubSystem):
        logging.debug("registering helicity angles in kinematics")
        if subsystem not in self._registered_subsystems:
            suffix = ''
            for fs in subsystem.final_states:
                suffix += '+'
                for x in fs:
                    suffix += str(x) + '_'
                suffix = suffix[:-1]
            if subsystem.recoil_state:
                suffix += '_vs_'
                for x in subsystem.recoil_state:
                    suffix += str(x) + '_'
                suffix = suffix[:-1]

            self._registered_subsystems[subsystem] = (
                'theta' + suffix, 'phi' + suffix)
        return self._registered_subsystems[subsystem]

    def register_subsystem(self, subsystem: SubSystem):
        state_fs = []
        for fs in subsystem.final_states:
            state_fs += fs
        invmass_name = self.register_invariant_mass(list(set(state_fs)))
        angle_names = self.register_helicity_angles(subsystem)

        return (invmass_name,) + angle_names

    def _convert_ids_to_indices(self, ids):
        if self._fs_id_event_pos_mapping:
            return [self._fs_id_event_pos_mapping[i] for i in ids]
        else:
            return ids

    def convert(self, events):
        logging.info('converting %s events', len(events[0]))

        dataset = {}

        for four_momenta_ids, inv_mass_name \
                in self._registered_inv_masses.items():
            four_momenta = np.sum(
                events[self._convert_ids_to_indices(four_momenta_ids), :],
                axis=0)

            dataset[inv_mass_name] = tfa_kin.mass_squared(
                np.array(four_momenta))

        for subsys, angle_names in self._registered_subsystems.items():
            topology = [
                np.sum(
                    events[self._convert_ids_to_indices(
                        subsys.final_states[0]), :],
                    axis=0),
                np.sum(
                    events[self._convert_ids_to_indices(
                        subsys.final_states[1]), :],
                    axis=0)
            ]
            if subsys.recoil_state:
                topology = [
                    topology,
                    np.sum(
                        events[self._convert_ids_to_indices(
                            subsys.recoil_state), :],
                        axis=0),
                ]
            if subsys.parent_recoil_state:
                topology = [
                    topology,
                    np.sum(
                        events[self._convert_ids_to_indices(
                            subsys.parent_recoil_state), :],
                        axis=0),
                ]

            values = tfa_kin.nested_helicity_angles(topology)

            # the last two angles is always what we are interested
            dataset[angle_names[0]] = values[-2].numpy()
            dataset[angle_names[1]] = values[-1].numpy()

        return dataset
