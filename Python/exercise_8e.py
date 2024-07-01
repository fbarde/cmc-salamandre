"""Exercise 8e"""

from cProfile import label
import os
import pickle
from turtle import speed
import numpy as np
from requests import head
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
from network import SalamandraNetwork
from salamandra_simulation.data import SalamandraData
import matplotlib.pyplot as plt
from scipy import integrate
from plot_results import plot_positions
from plot_results import plot_trajectory


#initial phases are set to random : 
def exercise_8e1(timestep):
    """Exercise 8e1"""
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=12,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=4,
            body_weights = 0,
            amp_first_last=[1.2, 1.5],
            gain_sensory = 0 # No sensory feedback
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
        data.to_file(filename.format(simulation_i, 'e1'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
    
    data = SalamandraData.from_file('logs/example/simulation_0.e1')
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

#initial phases are set in order to have a swimming behavior: 
def exercise_8e1_bis(timestep):
    """Exercise 8e1"""
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=12,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=4,
            amp_first_last=[1.2, 1.5],
            body_weights = 0,
            gain_sensory = 0 # No sensory feedback
        )
    ]

    # Grid search
    os.makedirs('./logs/example/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/example/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'water', give it a try!
            initial_phase=np.pi/8
            # fast=True,  # For fast mode (not real-time)
            # headless=True,  # For headless mode (No GUI, could be faster)
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'e1'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)

    data = SalamandraData.from_file('logs/example/simulation_0.e1')
    with open('logs/example/simulation_0.pickle', 'rb') as param_file:
        parameters = pickle.load(param_file)
    timestep = data.timestep
    n_iterations = np.shape(data.sensors.links.array)[0]
    network = SalamandraNetwork(parameters, n_iterations, data.state,initial_phase=np.pi/8)
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

    links_positions = data.sensors.links.urdf_positions()
    head_positions = links_positions[:, 0, :]
    tail_positions = links_positions[:, 8, :]
    joints_velocities = data.sensors.joints.velocities_all()
    joints_torques = data.sensors.joints.motor_torques_all()
    #print(links_positions[300, 0, 2])

    plt.figure()
    plot_trajectory(head_positions)
    plot_positions(times,head_positions)
    #plot_positions(times,tail_positions)
    plt.show()
    
    pass

   
#sweep on the sensory feedback coefficient : 
def exercise_8e2(timestep,several_gain=False):
    """Exercise 8e2"""
    
    gains = np.linspace(-50,50,80)
    # 2nd step : set total phase lags
    set_phase_lags = 2*np.pi/8

    if several_gain==True:
    # 3rd step : Loop x2 to make the grid of parameter_set
        parameter_set=[]
        for gainvalue in gains:
            sim_param = [SimulationParameters(
            duration=20,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=4,  # An example of parameter part of the grid search
            #amp_first_last=[2,2],  # Set the amplitude coeff
            turn=1,  # Another example
            gain_sensory = gainvalue,
            body_weights = 0,
            )]

            parameter_set = np.concatenate((parameter_set,sim_param))

    # PLOT THE RESULTS
    # Copy paste what is used in plot_results.py
    

        # Grid search
        os.makedirs('./logs/example/', exist_ok=True)
        for simulation_i, sim_parameters in enumerate(parameter_set):
            filename = './logs/example/simulation_{}.{}'
            sim, data = simulation(
                sim_parameters=sim_parameters,  # Simulation parameters, see above
                arena='water',  # Can also be 'water', give it a try! 'ground'
                fast=True,  # For fast mode (not real-time)
                headless=True,  # For headless mode (No GUI, could be faster)
            )
            # Log robot data
            data.to_file(filename.format(simulation_i, '8e2bis'), sim.iteration)
            # Log simulation parameters
            with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
                pickle.dump(sim_parameters, param_file)
    
    # Define energy and mean speed vectors
    N=len(gains)
    energy = np.zeros(N)
    mean_speed_z = np.zeros(N)
    COT = np.zeros(N)

    for i in range(N):
        # Load data
        data = SalamandraData.from_file(f'logs/example/simulation_{i}.8e2bis')
        with open(f'logs/example/simulation_{i}.pickle', 'rb') as param_file:
            parameters = pickle.load(param_file)
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
            start=0,
            stop=timestep*n_iterations,
            step=timestep,
        )
        timestep = times[1] - times[0]
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        tail_positions = links_positions[:, 8, :]
        joints_velocities = data.sensors.joints.velocities_all()
        joints_torques = data.sensors.joints.motor_torques_all()

        # Need to discard the initial transient : remove first 3s:
        idx_slice = int(3/timestep)
        head_positions = head_positions[idx_slice:]
        joints_velocities = joints_velocities[idx_slice:]
        joints_torques = joints_torques[idx_slice:]
        times = times[idx_slice:]

        #Compute the energy with sum 
        energy[i]= np.sum(np.multiply(joints_velocities, joints_torques)*timestep)

        #Compute speed as time take from starting point to end point
        head_pos_np = np.asarray(head_positions)
        mean_speed_z[i]= np.linalg.norm(head_pos_np[-1]-head_pos_np[0])/(times[-1]-times[0])

        #try with cost of transport. COT = E/(mgd)
        COT[i] =  energy[i]/np.linalg.norm(head_pos_np[-1]-head_pos_np[0])

    form1 = {'size': 14}

    plt.figure()
    plt.scatter(gains,energy,marker ="+")
    plt.xlabel("Gain sensory",fontdict=form1)
    plt.ylabel("Energy [J]",fontdict=form1)
    plt.title("Energy depending on gain sensory coefficient",fontdict=form1)
    plt.grid()
    plt.show()

    plt.figure()
    plt.scatter(gains,mean_speed_z,marker ="+")
    plt.xlabel("Gain sensory",fontdict=form1)
    plt.ylabel("Speed [m/s]",fontdict=form1)
    plt.title("speed depending on gain sensory coefficient",fontdict=form1)
    plt.grid()
    plt.show()

    plt.figure()
    plt.scatter(gains,COT,marker ="+")
    plt.yscale("log")
    plt.xlabel("Gain sensory",fontdict=form1)
    plt.ylabel("COT []",fontdict=form1)
    plt.title(" Cost of Transport depending on the gain sensory coefficient",fontdict=form1)
    plt.grid()
    plt.show()

    pass

#test on sensory feedback with wfb =-5
def exercise_8e2_bis(timestep):
   # Parameters
    parameter_set = [
        SimulationParameters(
            duration=12,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=4,
            body_weights = 0,
            gain_sensory = -5, # Sensory feedback
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
            #headless=True,  # For headless mode (No GUI, could be faster)
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'e2bis'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)

    data = SalamandraData.from_file('logs/example/simulation_0.e2bis')
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

    links_positions = data.sensors.links.urdf_positions()
    head_positions = links_positions[:, 0, :]
    tail_positions = links_positions[:, 8, :]
    joints_velocities = data.sensors.joints.velocities_all()
    joints_torques = data.sensors.joints.motor_torques_all()

    plt.figure()
    plot_trajectory(head_positions)
    plt.show()
    
    pass
   

if __name__ == '__main__':
    exercise_8e1(timestep=1e-2)
    exercise_8e1_bis(timestep=1e-2)
    exercise_8e2(timestep=1e-2,several_gain=True)
    exercise_8e2_bis(timestep=1e-2)
    
    


