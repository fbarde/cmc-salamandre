"""Exercise 8d"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
from salamandra_simulation.data import SalamandraData
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
from simulation_parameters import SimulationParameters
from network import SalamandraNetwork
from salamandra_simulation.data import SalamandraData
import matplotlib.pyplot as plt
from plot_results import plot_positions, plot_trajectory

def exercise_8d1(timestep):
    """Exercise 8d1"""
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=20,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=4,  # An example of parameter part of the grid search
            turn = 1.25,
            phase_lag=0,  # or np.zeros(n_joints) for example
            # ...
        )
    ]

    # Grid search
    os.makedirs('./logs/example/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/example/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'water', give it a try!
            # fast=True,  # For fast mode (not real-time)
            # headless=True,  # For headless mode (No GUI, could be faster)
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, '8d1'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
    
    pass


def exercise_8d2(timestep):
    """Exercise 8d2"""
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=20,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=4,  # An example of parameter part of the grid search
            phase_bias_body_up = -np.pi/4,
            phase_bias_body_down = np.pi/4,
            phase_lag=0,  # or np.zeros(n_joints) for example
            # ...
        )
    ]

    # Grid search
    os.makedirs('./logs/example/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/example/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'water', give it a try!
            # fast=True,  # For fast mode (not real-time)
            # headless=True,  # For headless mode (No GUI, could be faster)
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, '8d2'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
    pass

def plot_traj(nb, plot=True):
    # Load data
    data = SalamandraData.from_file('logs/example/simulation_0.8d{}'.format(nb))
    with open('logs/example/simulation_0.pickle', 'rb') as param_file:
        parameters = pickle.load(param_file)
    timestep = data.timestep
    n_iterations = np.shape(data.sensors.links.array)[0]
    times = np.arange(
        start=0,
        stop=timestep*n_iterations,
        step=timestep,
    )
    timestep = times[1] - times[0]
    #amplitudes = parameters.amplitudes
    phase_lag = parameters.phase_lag
    osc_phases = data.state.phases()
    osc_amplitudes = data.state.amplitudes()
    links_positions = data.sensors.links.urdf_positions()
    head_positions = links_positions[:, 0, :]
    tail_positions = links_positions[:, 8, :]
    joints_positions = data.sensors.joints.positions_all()
    joints_velocities = data.sensors.joints.velocities_all()
    joints_torques = data.sensors.joints.motor_torques_all()

    # Notes:
    # For the links arrays: positions[iteration, link_id, xyz]
    # For the positions arrays: positions[iteration, xyz]
    # For the joints arrays: positions[iteration, joint]

    # Plot data
    plt.figure('Positions')
    plot_positions(times, head_positions)
    plt.figure('Trajectory')
    plot_trajectory(head_positions)
    plt.figure('Spine angles')
    plot_trajectory(joints_positions)
    
    #plot_oscillations()
    
    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()

def plot_osc_stacked(nb):
    data = SalamandraData.from_file('logs/example/simulation_0.8d{}'.format(nb))
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

    x = np.zeros([len(times),parameters.n_body_joints*2+parameters.n_legs_joints])
    for i in range(0,parameters.n_body_joints*2):
        x[:,i] = network.state.amplitudes()[:,i]*(1+np.cos(data.state.phases()[:,i]))

    for j in range(parameters.n_body_joints*2,parameters.n_body_joints*2+parameters.n_legs_joints ):
        x[:,j] = data.state.amplitudes()[:,j]*(1+np.cos(data.state.phases()[:,j]))

    
    fig, ax = plt.subplots()
    for i in range(0,parameters.n_body_joints) : 
        if i==0 : 
            ax.plot(times,np.add(x[:,i], np.pi*i/3), color='tab:blue',label = 'left oscillator')
            ax.plot(times,np.add(x[:,i+8], np.pi*i/3), color='tab:orange',label = 'right oscillator')
            ax.text(duration,np.pi*i/3+0.2, "x{},x{}".format(i+1,i+9), horizontalalignment='center',size='large')
        else : 
            ax.plot(times,np.add(x[:,i], np.pi*i/3), color='tab:blue')
            ax.plot(times,np.add(x[:,i+8], np.pi*i/3), color='tab:orange')
            ax.text(duration,np.pi*i/3+0.2, "x{},x{}".format(i+1,i+9), horizontalalignment='center',size='large')
    plt.legend(loc=2)
    plt.title('Oscillator patterns for the body oscillators',size = 14)
    plt.xlabel('Time [s]',size = 13)
    plt.ylabel('Joint number',size = 13)
    plt.show()

    links_positions = data.sensors.links.urdf_positions()
    head_positions = links_positions[:, 0, :]
    tail_positions = links_positions[:, 8, :]
    joints_velocities = data.sensors.joints.velocities_all()
    joints_torques = data.sensors.joints.motor_torques_all()

    plt.figure()
    plot_trajectory(head_positions)
    plt.show()

def plot_osc(nb):
    data = SalamandraData.from_file('logs/example/simulation_0.8d{}'.format(nb))
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

    x = np.zeros([len(times),parameters.n_body_joints*2+parameters.n_legs_joints])
    for i in range(0,parameters.n_body_joints*2):
        x[:,i] = network.state.amplitudes()[:,i]*(1+np.cos(data.state.phases()[:,i]))

    for j in range(parameters.n_body_joints*2,parameters.n_body_joints*2+parameters.n_legs_joints ):
        x[:,j] = data.state.amplitudes()[:,j]*(1+np.cos(data.state.phases()[:,j]))

    form1 = {'size': 13}

    #oscillators
    fig, ax = plt.subplots()
    for i in range(0,4):
        ax.plot(times,np.add(x[:,i], np.pi*i/3), color='tab:blue')
        ax.text(duration-1,x[int((duration-1)/timestep),i]+np.pi*i/3+0.2, "x{}".format(i+1), horizontalalignment='center',size='large')
    for i in range(4,8):
        ax.plot(times,np.add(x[:,i], np.pi*i/3), color='tab:orange')
        ax.text(duration-1,x[int((duration-1)/timestep),i]+np.pi*i/3+0.2, "x{}".format(i+1), horizontalalignment='center',size='large')
    for i in range(8,12):
        ax.plot(times,np.add(x[:,i], np.pi*i/3), color='tab:blue')
        ax.text(duration-1,x[int((duration-1)/timestep),i]+np.pi*i/3+0.2, "x{}".format(i+1), horizontalalignment='center',size='large')
    for i in range(12,16):
        ax.plot(times,np.add(x[:,i], np.pi*i/3), color='tab:orange')
        ax.text(duration-1,x[int((duration-1)/timestep),i]+np.pi*i/3+0.2, "x{}".format(i+1), horizontalalignment='center',size='large')

   
    ax.set_title('Oscillator patterns for the body oscillators',size = 14)
    ax.set_xlabel('Time [s]',form1)
    ax.set_ylabel("x body",form1)
    ax.set_yticks([])
    
    plt.grid(axis='y')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    exercise_8d1(timestep=1e-2)
    #exercise_8d2(timestep=1e-2)

    plot_traj(1)
    #plot_traj(2)
    plot_osc_stacked(1)
