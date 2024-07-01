"""Network controller"""

import numpy as np
from farms_core.model.control import AnimatController


class SalamandraController(AnimatController):
    """Salamandra controller"""

    def __init__(self, joints_names, animat_data, network,arena='water'):
        super().__init__(
            joints_names=[joints_names, [], []],
            max_torques=[np.ones(len(joints_names)), [], []],
        )
        self.network = network
        self.animat_data = animat_data
        self.arena = arena


    def step(self, iteration, time, timestep):
        """Control step"""
        # Lateral hydrodynamic forces
        loads = -1 * np.concatenate([
            self.animat_data.sensors.xfrc.array[
                iteration,
                0:self.network.robot_parameters['n_body_joints'],
                1
            ],
            np.zeros(self.network.robot_parameters['n_legs_joints'])
        ])
        # Provide salamanrdra data to robot parameters
        if self.arena == 'amphibious' :
            self.network.robot_parameters.step(
                iteration=iteration,
                salamandra_data=self.animat_data,
            )
        # Network integration step
        self.network.step(iteration, time, timestep, loads)

    def positions(self, iteration, time, timestep):
        """Postions"""
        return dict(zip(
            self.joints_names[0],
            self.network.get_motor_position_output(iteration),
        ))

