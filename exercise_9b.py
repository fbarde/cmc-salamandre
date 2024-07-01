"""Exercise 9b"""

import numpy as np
from farms_core import pylog
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
from network import SalamandraNetwork
from salamandra_simulation.data import SalamandraData
import os
import pickle
import matplotlib.pyplot as plt
from plot_results import plot_positions

#if water to land is True, the robot starts swimming then walking, if False, it starts walking then swimming.
def exercise_9b(timestep,water_to_land = False):
    """Exercise example"""

    if water_to_land == False : 
        spawn_position=[0, 0, 0.1]  # Robot position in [m]
        spawn_orientation=[0, 0, 0]  # Orientation in Euler angles [rad]
        record_path_opt='walk2swim'
    else : 
        spawn_position=[4, 0, 0.0]
        spawn_orientation=[0, 0, np.pi]
        record_path_opt='swim2walk'


    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=20,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=spawn_position,  # Robot position in [m]
            spawn_orientation=spawn_orientation,  # Orientation in Euler angles [rad]
            drive=4,  # An example of parameter part of the grid search
            phase_bias_limb_body = 5*np.pi/3,
            nom_amp0_body = 0.2,
            nom_amp_drive_body = 0.1,
        )
    ]

    # Grid search
    os.makedirs('./logs/example/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/example/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            #fast=True,  # For fast mode (not real-time)
            #headless=True,  # For headless mode (No GUI, could be faster)
            arena='amphibious',
            record=True,
            record_path=record_path_opt,  # or swim2walk
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, '9b1'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)

    data = SalamandraData.from_file('logs/example/simulation_0.9b1')
    with open('logs/example/simulation_0.pickle', 'rb') as param_file:
        parameters = pickle.load(param_file)
    timestep = data.timestep
    n_iterations = np.shape(data.sensors.links.array)[0]
    network = SalamandraNetwork(parameters, n_iterations, data.state)
    times = np.arange(
        start=0,
        stop=timestep*n_iterations,
        step=timestep,
    )
    timestep = times[1] - times[0]
    duration = parameters.duration

    links_positions = data.sensors.links.urdf_positions()
    head_positions = links_positions[:, 0, :]
    tail_positions = links_positions[:, 8, :]
    limb_positions1 = links_positions[:, 9, :]
    limb_positions2 = links_positions[:, 10, :]
    limb_positions3 = links_positions[:, 11, :]
    limb_positions4 = links_positions[:, 12, :]
    #if you want the value as numpy array do : np.array(link_position[:, 10, :])
    joints_velocities = data.sensors.joints.velocities_all()
    joints_torques = data.sensors.joints.motor_torques_all()

    plot_positions(times,head_positions)
    plt.show()

    plt.figure()
    legend =[]
    for i in range(9): 
        plt.plot(times,links_positions[:, i, 0],lw=2)
        legend.append("link : " + str(i))
    plt.legend(legend)
    plt.xlabel("Time [s]",size=13)
    plt.ylabel("X coordinate [m]",size=13)
    plt.title("X coordinate from GPS signal for the 9 links",size=14)
    plt.grid()
    plt.show()

    spine_angle = np.zeros([len(times),parameters.n_body_joints])
    limb_angle = np.zeros([len(times),parameters.n_legs_joints])
    for i in range(0,parameters.n_body_joints):
        for j in range(len(times)):
            if network.state.amplitudes()[j,17]>0:
                spine_angle[j,i] = 0.7*(network.state.amplitudes()[j,i]*(1+np.cos(data.state.phases()[j,i]))-network.state.amplitudes()[j,i+8]*(1+np.cos(data.state.phases()[j,i+8])))
            else :
                spine_angle[j,i] = network.state.amplitudes()[j,i]*(1+np.cos(data.state.phases()[j,i]))-network.state.amplitudes()[j,i+8]*(1+np.cos(data.state.phases()[j,i+8]))

    for j in range(0,parameters.n_legs_joints):
        limb_angle[:,j] = data.state.phases()[:,j+16]-np.pi/2


    fig, ax = plt.subplots(1,1,sharex=True,sharey=True)
    for i in range(0,8):
        ax.plot(times,np.add(spine_angle[:,i], np.pi*i/3), color='tab:blue')
        ax.text(duration+1,spine_angle[int((duration-1)/timestep),i]+np.pi*i/3+0.2, "Joint {}".format(i+1), horizontalalignment='center',size='large')

    plt.grid()
    plt.xlabel("Time [s]",size=13)
    plt.ylabel("Spinal joint angle [Rad]",size=13)
    plt.title("Motor outputs of the different spinal joints ",size=14)
    plt.show()


if __name__ == '__main__':
    exercise_9b(timestep=1e-2,water_to_land=True)

