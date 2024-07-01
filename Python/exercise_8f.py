"""Exercise 8f"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
from salamandra_simulation.data import SalamandraData
from plot_results import plot_2d
from scipy import integrate


def exercise_8f(timestep):
    """Exercise 8f"""
     # Parameters
    # PERFORM THE GRID SEARCH 
    # 1st step: Create a vector with different set of couplings weights
    set_couplings = np.linspace(0, 20, 10)
    # 2nd step : Create a vector w/ different set of sensory gain
    set_sensory_feed = -1*np.linspace(0, 7, 10)

# 3rd step : Loop x2 to make the grid of parameter_set
    parameter_set=[]
    for coupling in set_couplings:
        for w_fb in set_sensory_feed:
            sim_param = [SimulationParameters(
            duration=30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=4,  # An example of parameter part of the grid search
            amp_first_last=[1.2, 1.5],  # Set the amplitude gradient
            phase_bias_body_up = 1.1,
            phase_bias_body_down = -1.1,
            body_weights = coupling,
            gain_sensory = w_fb
            )]

            parameter_set = np.concatenate((parameter_set,sim_param))


    # Grid search
    os.makedirs('./logs/8f/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/8f/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'water', give it a try! 'ground'
            fast=True,  # For fast mode (not real-time)
            headless=True,  # For headless mode (No GUI, could be faster)
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, '8f'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)

    


def plot_8f(timestep):
    """Exercise 8f: PLOT 3D PLOT FOR GRID SEARCH"""
    # PLOT THE RESULTS
    
    # Define energy, mean speed  and COT vectors
    N=10*10
    energy = np.zeros(N) # Compute energy w/ sum
    energies = np.zeros(N) # Compute enery w/ integral
    mean_speed = np.zeros(N)
    couplings_grid = np.zeros(N)
    sensory_feed_grid = np.zeros(N)
    COT =  np.zeros(N) # Cost of transport


    for i in range(N):
        # Load data
        data = SalamandraData.from_file(f'logs/8f/simulation_{i}.8f')
        with open(f'logs/8f/simulation_{i}.pickle', 'rb') as param_file:
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
        couplings_grid[i] = parameters.body_weights
        sensory_feed_grid[i] = parameters.gain_sensory

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

        
        # Compute the Cost of Transport:
        COT[i] =  energies[i]/np.linalg.norm(head_pos_np[-1]-head_pos_np[0])
    
    
    grid_energies= np.stack((couplings_grid,sensory_feed_grid,energies), axis=1)
    plt.figure('Grid search energy intg')
    plot_2d(grid_energies,labels=('Coupling weights [ ]', 'Sensory gain [ ]', 'Energy [J]'),cmap='nipy_spectral')

    grid_mean_speed= np.stack((couplings_grid,sensory_feed_grid, mean_speed), axis=1)
    plt.figure('Grid search mean speed')
    plot_2d(grid_mean_speed,labels=('Coupling weights [ ]', 'Sensory gain [ ]', r'Mean Speed  [ms$^{-1}$]'),cmap='nipy_spectral') #,cmap='hsv'

    grid_COT= np.stack((couplings_grid,sensory_feed_grid, COT), axis=1)
    plt.figure('Grid search cost of transport')
    plot_2d(grid_COT,labels=('Coupling weights [ ]', 'Sensory gain [ ]', 'Cost of transport  [ ]'),cmap='nipy_spectral') #,cmap='hsv'


    plt.show()



if __name__ == '__main__':
    exercise_8f(timestep=1e-2) #Do the grid search
    plot_8f(timestep=1e-2) #Plot the results of grid search

