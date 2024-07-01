"""Exercise 8b"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
from salamandra_simulation.data import SalamandraData
from plot_results import plot_2d
from plot_results import plot_positions
from scipy import integrate


def exercise_8b(timestep, grid_search=False):
    """Exercise 8b"""
 # Parameters
   
    # PERFORM THE GRID SEARCH 
    # 1st step: Create a vector with different set of amplitudes 
    set_amplitudes = np.linspace(0, 2, 10)
    # 2nd step : Create a vector w/ different set of phase lags
    set_phase_lags = np.pi * np.linspace(0, 3.2, 10)/8

    if grid_search==True:
    # 3rd step : Loop x2 to make the grid of parameter_set
        parameter_set=[]
        for amp in set_amplitudes:
            for lag in set_phase_lags:
                sim_param = [SimulationParameters(
                duration=30,  # Simulation duration in [s]
                timestep=timestep,  # Simulation timestep in [s]
                spawn_position=[0, 0, 0.1],  # Robot position in [m]
                spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
                drive=4,  # An example of parameter part of the grid search
                amp_first_last=[amp, amp],  # Set the amplitude coeff
                phase_bias_body_up = lag,
                phase_bias_body_down = -lag
                )]

                parameter_set = np.concatenate((parameter_set,sim_param))

    

        # Grid search
        os.makedirs('./logs/8b/', exist_ok=True)
        for simulation_i, sim_parameters in enumerate(parameter_set):
            filename = './logs/8b/simulation_{}.{}'
            sim, data = simulation(
                sim_parameters=sim_parameters,  # Simulation parameters, see above
                arena='water',  # Can also be 'water', give it a try! 'ground'
                fast=True,  # For fast mode (not real-time)
                headless=True,  # For headless mode (No GUI, could be faster)
            )
            # Log robot data
            data.to_file(filename.format(simulation_i, '8b'), sim.iteration)
            # Log simulation parameters
            with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
                pickle.dump(sim_parameters, param_file)

    # PLOT THE RESULTS
    
    # Define energy and mean speed vectors
    N=len(set_amplitudes)*len(set_phase_lags)
    energy = np.zeros(N) # Compute energy w/ sum
    energies = np.zeros(N) # Compute enery w/ integral
    mean_speed = np.zeros(N)
    mean_speed2 = np.zeros(N)
    amplitudes_grid = np.zeros(N)
    phase_lags_grid = np.zeros(N)
    COT =  np.zeros(N) # Cost of transport


    for i in range(N):
        # Load data
        data = SalamandraData.from_file(f'logs/8b/simulation_{i}.8b')
        with open(f'logs/8b/simulation_{i}.pickle', 'rb') as param_file:
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
        joints_velocities = data.sensors.joints.velocities_all()
        joints_torques = data.sensors.joints.motor_torques_all()
        amplitudes_grid[i] = parameters.amp_first_last[0]
        phase_lags_grid[i] = parameters.phase_bias_body_up

        # Need to discard the initial transient : remove first 3s:
        idx_slice = int(3/timestep)
        head_positions = head_positions[idx_slice:]
        joints_velocities = joints_velocities[idx_slice:]
        joints_torques = joints_torques[idx_slice:]
        times = times[idx_slice:]


        #Compute the energy with sum 
        energy[i]= np.sum(np.multiply(joints_velocities, joints_torques)*timestep)

        #Compute the energy with integral 
        for j in range(12): #np.shape(joints_velocities)[1]
            values2int = np.multiply(joints_velocities[:,j],joints_torques[:,j])
            energies[i] += integrate.simps(values2int,times)



        #Compute speed as time take from starting point to end point
        head_pos_np = np.asarray(head_positions)
        mean_speed[i]= np.linalg.norm(head_pos_np[-1]-head_pos_np[0])/(times[-1]-times[0])

        # Compute mean speed : NOT USED 
        speed_dt =0
        for j in range(len(head_pos_np)-1):
            speed_dt += np.linalg.norm(head_pos_np[j+1]-head_pos_np[j])/timestep

        mean_speed2[i] = speed_dt/(len(head_pos_np)-1)

        # Compute the Cost of Transport:
        COT[i] =  energies[i]/np.linalg.norm(head_pos_np[-1]-head_pos_np[0])
    

    grid_energies= np.stack((amplitudes_grid,phase_lags_grid,energies), axis=1)
    plt.figure('Grid search energy intg')
    plot_2d(grid_energies,labels=('Amplitude [ ]', 'Phase Lag [rad]', 'Energy [J]'),cmap='nipy_spectral')

    grid_mean_speed= np.stack((amplitudes_grid, phase_lags_grid, mean_speed), axis=1)
    plt.figure('Grid search mean speed')
    plot_2d(grid_mean_speed,labels=('Amplitude [ ]', 'Phase Lag [rad]', r'Mean Speed  [ms$^{-1}$]'),cmap='nipy_spectral') #,cmap='hsv'
    

    grid_COT= np.stack((amplitudes_grid, phase_lags_grid, COT), axis=1)
    plt.figure('Grid search cost of transport')
    plot_2d(grid_COT,labels=('Amplitude [ ]', 'Phase Lag [rad]', 'Cost of transport  [ ]'),cmap='nipy_spectral',log=True) #,cmap='hsv'
    
    # Doesn't show any interesting behavior...

    grid_mean_speed2= np.stack((amplitudes_grid, phase_lags_grid, mean_speed2), axis =1)
    plt.figure('Grid search mean speed2')
    plot_2d(grid_mean_speed2,labels=('Amplitude [ ]', 'Phase Lag [rad]', r'Mean Speed  [ms$^{-1}$]'),cmap='nipy_spectral')
    #plt.show()

    #plt.figure('Plot positions')
    #plot_positions(times, head_positions)

    plt.show()




def exercise_8b_bonus(timestep):
    """Exercise 8b: Function used to see the effect of the different parameters"""
    # Parameters

    parameter_set = [
        SimulationParameters(
            duration=40,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=4,  # An example of parameter part of the grid search
            amp_first_last=[1.3, 1.3],  # Set the amplitude coeff
            phase_bias_body_up = 1.1,
            phase_bias_body_down = -1.1
        )
    ]

    os.makedirs('./logs/8b_bonus/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/8b_bonus/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'water', give it a try!
            # fast=True,  # For fast mode (not real-time)
            # headless=True,  # For headless mode (No GUI, could be faster)
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, '8b2'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)


if __name__ == '__main__':
    exercise_8b(timestep=1e-2, grid_search=False)
    exercise_8b_bonus(timestep=1e-2)

