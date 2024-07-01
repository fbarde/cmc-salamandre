"""Exercise 8c"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
from salamandra_simulation.data import SalamandraData
from plot_results import plot_2d
from plot_results import plot_positions
from plot_results import plot_trajectory
from scipy import integrate

# grid search on the gradient amplitude 
def exercise_8c(timestep,grid_search=False):
    """Exercise 8c"""
 # Parameters
   
    # PERFORM THE GRID SEARCH 
    # 1st step: Create a vector with different set of amplitudes for R_head
    set_amplitudes_rhead = np.linspace(0.1, 2, 10)
    # 1st step: Create a vector with different set of amplitudes for R_tail
    set_amplitudes_rtail = np.linspace(0.1, 2, 10)
    # 2nd step : set total phase lags
    set_phase_lags = 2*np.pi/8

    if grid_search==True:
    # 3rd step : Loop x2 to make the grid of parameter_set
        parameter_set=[]
        for amp_head in set_amplitudes_rhead:
            for  amp_tail in set_amplitudes_rtail:
                sim_param = [SimulationParameters(
                duration=20,  # Simulation duration in [s]
                timestep=timestep,  # Simulation timestep in [s]
                spawn_position=[0, 0, 0.1],  # Robot position in [m]
                spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
                drive=4,  # An example of parameter part of the grid search
                amp_first_last=[amp_head, amp_tail],  # Set the amplitude coeff
                phase_bias_body_up = set_phase_lags,
                phase_bias_body_down = -set_phase_lags,
                turn=1,  # Another example
                freq0_body=1 
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
            data.to_file(filename.format(simulation_i, '8c'), sim.iteration)
            # Log simulation parameters
            with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
                pickle.dump(sim_parameters, param_file)
    
    # Define energy and mean speed vectors
    N=len(set_amplitudes_rhead)*len(set_amplitudes_rtail)

    energy = np.zeros(N)
    energies = np.zeros(N) #try integral thing
    mean_speed = np.zeros(N)
    mean_speed_z = np.zeros(N)
    mean_speed2 = np.zeros(N)
    amplitudes_head_grid = np.zeros(N)
    amplitudes_tail_grid = np.zeros(N)
    COT = np.zeros(N)
    #phase_lags_grid = np.zeros(N)

    for i in range(N):
        # Load data
        data = SalamandraData.from_file(f'logs/example/simulation_{i}.8c')
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
        amplitudes_head_grid[i] = parameters.amp_first_last[0]
        amplitudes_tail_grid[i] = parameters.amp_first_last[1]

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

    # 2D plots
    grid_energy= np.stack((amplitudes_head_grid,amplitudes_tail_grid,energy), axis=1)
    plt.figure('Grid search energy')
    plot_2d(grid_energy,labels=('R head', 'R tail', 'Energy'),cmap = "nipy_spectral")
    #plt.show()

    grid_mean_speed_z= np.stack((amplitudes_head_grid,amplitudes_tail_grid, mean_speed_z), axis=1)
    plt.figure('Grid search mean speed')
    plot_2d(grid_mean_speed_z,labels=('R head', 'R tail', 'Mean Speed'),cmap = "nipy_spectral")
    #plt.show()

    grid_COT= np.stack((amplitudes_head_grid,amplitudes_tail_grid, COT), axis=1)
    plt.figure('Grid COT')
    plot_2d(grid_COT,labels=('R head', 'R tail', 'Cost of Transport'), cmap = "nipy_spectral")

    plt.figure('position head')
    plot_positions(times, head_positions)

    plt.figure('position tail')
    plot_positions(times, tail_positions)

    plt.figure('position tail')
    plot_positions(times, tail_positions)

    plt.show()

# function used for the body deformations
def exercise_8c_simulation(timestep):
    # Parameters
    # 2nd step : set total phase lags
    set_phase_lags = 2*np.pi/8
    parameter_set = [
        SimulationParameters(
            duration= 10, #20,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=4,  # An example of parameter part of the grid search
            amp_first_last=[1.2, 1.5],  # Set the amplitude gradient
            phase_bias_body_up = set_phase_lags,
            phase_bias_body_down = -set_phase_lags,
            turn=1,  # Another example
            freq0_body=1 
        )
    ]

    os.makedirs('./logs/8c_simulation/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/8c_simulation/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'water', give it a try!
            #fast=True,  # For fast mode (not real-time)
            #headless=True,  # For headless mode (No GUI, could be faster)
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, '8c2'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)


    # PLOT THE JOINT POSITIONS :
    N=1

    for i in range(N):
        # Load data
        data = SalamandraData.from_file(f'logs/8c_simulation/simulation_{i}.8c2')
        with open(f'logs/8c_simulation/simulation_{i}.pickle', 'rb') as param_file:
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
        #head_positions = links_positions[:, 0, :]
        #joints_velocities = data.sensors.joints.velocities_all()
        #joints_torques = data.sensors.joints.motor_torques_all()
        

        # Need to discard the initial transient : remove first 3s:
        idx_slice = int(3.4/timestep)
        links_positions = links_positions[idx_slice:]
        #head_positions = head_positions[idx_slice:]
        #joints_velocities = joints_velocities[idx_slice:]
        #joints_torques = joints_torques[idx_slice:]
        times = times[idx_slice:]
        #links_positions_np = np.asarray(links_positions)
        
        num_idx = int(170e-3/timestep)

        num_plots=8
        pos_links = np.zeros((num_plots,9,2)) #Shape : 9 plots x 9 links x 2 coord (x and y)

        fig, ax = plt.subplots(nrows=8, figsize=(6, 25))


        for i in range(num_plots):

            pos_links[i]= links_positions[i*num_idx, 0:9, 0:2]
            time = format(times[i*num_idx] , '.2f')
            ax[i].plot(pos_links[i,:,0], pos_links[i,:,1], 's-', lw=2, label=f'{time} s')
            ax[i].vlines(x=pos_links[0,0,0], ymin= min(pos_links[i,:,1])-0.01, ymax=max(pos_links[i,:,1])+0.01, color='red', linestyles='dashed')
            ax[i].grid()
            
            ax[i].legend(loc='upper right')

        ax[7].set_xlabel('x [m]')
        ax[4].set_ylabel('y [m]')
        plt.suptitle('Body deformations (links positions) of the salameter at different times.')
        plt.show()


if __name__ == '__main__':
    exercise_8c_simulation(timestep=1e-2)
    exercise_8c(timestep=1e-2,grid_search=False)
