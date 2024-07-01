"""Exercise 8g"""

import os
import pickle
import random
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
from salamandra_simulation.data import SalamandraData
from scipy import integrate
import matplotlib.pyplot as plt

## 8g1

#for this question we have 3 systems :  CPG - only segmental oscillators and intersegmental coupling, 
# decoupled - segmental oscillators and sensory feedback but no intersegmental coupling, 
# combined - CPG and sensory feedback.
# the 4 different disruptions applied are : 
# muted sensors : setting the respectivefeedback strength to zero (wfb = 0).
# removing couplings : setting the respective coupling weights to zero (wnj = 0).
# muted oscillators : implemented by setting the corresponding intrinsic frequencies of oscillators to zero (f = 0)
# mixed disruptions : all 3 together, or juste 2 ?

def exercise_8g1(timestep):
    """Exercise 8g1: Disruptions on CPG network"""

    # Use a vector of seed to make 10 different simulations and then average over it
    seed = np.arange(1, 11, dtype=int)
    print(seed)


    for seed_ in seed:
        for type_disrpt in ['sensors', 'oscillators', 'couplings','mixed']:
            if (type_disrpt=='couplings'): ndisrpt=8 
            else: ndisrpt=9 
            for nb_disrpt in range(ndisrpt): 

                if(type_disrpt=='sensors'):
                    # DISRUPTION: MUTED SENSORS
                    sim_parameters = SimulationParameters(
                            duration=30,  # Simulation duration in [s]
                            timestep=timestep,  # Simulation timestep in [s]
                            spawn_position=[0, 0, 0.1],  # Robot position in [m]
                            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
                            drive=4,  # An example of parameter part of the grid search
                            amp_first_last= [0.5, 0.5], #[0.90, 1.5],  # Set the amplitude gradient #[1.25]
                            phase_bias_body_up = 1.1,
                            phase_bias_body_down = -1.1,
                            muted_sensors = nb_disrpt,
                            seed_disruption = seed_
                        )
                
                if(type_disrpt=='oscillators'):
                    # DISRUPTION: MUTED OSCILLATORS
                    sim_parameters = SimulationParameters(
                            duration=30,  # Simulation duration in [s]
                            timestep=timestep,  # Simulation timestep in [s]
                            spawn_position=[0, 0, 0.1],  # Robot position in [m]
                            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
                            drive=4,  # An example of parameter part of the grid search
                            amp_first_last= [0.5, 0.5],#[0.90, 1.5],  # Set the amplitude gradient
                            phase_bias_body_up = 1.1,
                            phase_bias_body_down = -1.1,
                            muted_oscillators  = nb_disrpt,
                            seed_disruption = seed_,
                        )
                
                if(type_disrpt=='couplings'):
                    # DISRUPTION: MUTED COUPLINGS
                    sim_parameters = SimulationParameters(
                            duration=30,  # Simulation duration in [s]
                            timestep=timestep,  # Simulation timestep in [s]
                            spawn_position=[0, 0, 0.1],  # Robot position in [m]
                            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
                            drive=4,  # An example of parameter part of the grid search
                            amp_first_last= [0.5, 0.5],#[0.90, 1.5],  # Set the amplitude gradient
                            phase_bias_body_up = 1.1,
                            phase_bias_body_down = -1.1,
                            remove_couplings = nb_disrpt,
                            seed_disruption = seed_
                        )

                if(type_disrpt=='mixed'):
                    # DISRUPTION: MIXED DISRUPTIONS
                    # Choose randomly a certain number of disruptions for each type
                    mix_disrpt= [0,0,0]
                    random.seed(6)
                    mix_disrpt[0] = int(np.squeeze(np.random.randint(0,(nb_disrpt+1),1)))
                    mix_disrpt[1] = int(np.squeeze(np.random.randint(0,(nb_disrpt+1-mix_disrpt[0]),1)))
                    mix_disrpt[2] = nb_disrpt-mix_disrpt[0]-mix_disrpt[1]
                    if mix_disrpt[2]==8: mix_disrpt[2]=7
                    np.squeeze(mix_disrpt)
                    print(mix_disrpt)
                    

                    sim_parameters = SimulationParameters(
                            duration=30,  # Simulation duration in [s]
                            timestep=timestep,  # Simulation timestep in [s]
                            spawn_position=[0, 0, 0.1],  # Robot position in [m]
                            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
                            drive=4,  # An example of parameter part of the grid search
                            amp_first_last= [0.5, 0.5],#[0.90, 1.5],  # Set the amplitude gradient
                            phase_bias_body_up = 1.1,
                            phase_bias_body_down = -1.1,
                            muted_sensors = mix_disrpt[0],
                            muted_oscillators  = mix_disrpt[1],
                            remove_couplings = mix_disrpt[2],
                            seed_disruption = seed_
                        )


                # Save the simulation datas
                os.makedirs('./logs/8g_CPG/', exist_ok=True)
                filename = './logs/8g_CPG/simulation_{}_seed{}.{}'
                sim, data = simulation(
                    sim_parameters=sim_parameters,  # Simulation parameters, see above
                    arena='water',  # Can also be 'water', give it a try!
                    fast=True,  # For fast mode (not real-time)
                    headless=True,  # For headless mode (No GUI, could be faster)
                )
                # Log robot data
                data.to_file(filename.format(nb_disrpt, seed_ , type_disrpt), sim.iteration)
                # Log simulation parameters
                with open(filename.format(nb_disrpt, seed_ ,'pickle'), 'wb') as param_file:
                    pickle.dump(sim_parameters, param_file)


def exercise_8g2(timestep):
    """Exercise 8g2: Disruptions on Decoupled segment network"""
    # Network with decoupled segments AND sensory feedback

    seed = np.arange(1, 11, dtype=int)

    for seed_ in seed:
        for type_disrpt in ['sensors', 'oscillators', 'couplings', 'mixed']:
            if (type_disrpt=='couplings'): ndisrpt=8 
            else: ndisrpt=9 
            for nb_disrpt in range(ndisrpt): 

                if(type_disrpt=='sensors'):
                    # DISRUPTION: MUTED SENSORS
                    sim_parameters = SimulationParameters(
                            duration=30,  # Simulation duration in [s]
                            timestep=timestep,  # Simulation timestep in [s]
                            spawn_position=[0, 0, 0.1],  # Robot position in [m]
                            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
                            drive=4,  # An example of parameter part of the grid search
                            amp_first_last=[0.90, 1.5],  # Set the amplitude gradient
                            phase_bias_body_up = 1.1,
                            phase_bias_body_down = -1.1,
                            body_weights = 0, # Decoupled segments
                            gain_sensory = -2, # Sensory feedback
                            muted_sensors = nb_disrpt,
                            seed_disruption = seed_
                        )
                
                if(type_disrpt=='oscillators'):
                    # DISRUPTION: MUTED OSCILLATORS
                    sim_parameters = SimulationParameters(
                            duration=30,  # Simulation duration in [s]
                            timestep=timestep,  # Simulation timestep in [s]
                            spawn_position=[0, 0, 0.1],  # Robot position in [m]
                            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
                            drive=4,  # An example of parameter part of the grid search
                            amp_first_last=[0.90, 1.5],  # Set the amplitude gradient
                            phase_bias_body_up = 1.1,
                            phase_bias_body_down = -1.1,
                            body_weights = 0, # Decoupled segments
                            gain_sensory = -2, # Sensory feedback
                            muted_oscillators  = nb_disrpt,
                            seed_disruption = seed_
                        )
                
                if(type_disrpt=='couplings'):
                    # DISRUPTION: MUTED COUPLINGS
                    sim_parameters = SimulationParameters(
                            duration=30,  # Simulation duration in [s]
                            timestep=timestep,  # Simulation timestep in [s]
                            spawn_position=[0, 0, 0.1],  # Robot position in [m]
                            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
                            drive=4,  # An example of parameter part of the grid search
                            amp_first_last=[0.90, 1.5],  # Set the amplitude gradient
                            #phase_bias_body_up = 1.1,
                            #phase_bias_body_down = -1.1,
                            body_weights = 0, # Decoupled segments
                            gain_sensory = -2, # Sensory feedback
                            remove_couplings = nb_disrpt,
                            seed_disruption = seed_
                        )

                if(type_disrpt=='mixed'):
                    # DISRUPTION: MIXED DISRUPTIONS
                    mix_disrpt= [0,0,0]
                    random.seed(6)
                    mix_disrpt[0] = int(np.squeeze(np.random.randint(0,(nb_disrpt+1),1)))
                    mix_disrpt[1] = int(np.squeeze(np.random.randint(0,(nb_disrpt+1-mix_disrpt[0]),1)))
                    mix_disrpt[2] = nb_disrpt-mix_disrpt[0]-mix_disrpt[1]
                    if mix_disrpt[2]==8: mix_disrpt[2]=7
                    np.squeeze(mix_disrpt)
                    print(mix_disrpt)

                    sim_parameters = SimulationParameters(
                            duration=30,  # Simulation duration in [s]
                            timestep=timestep,  # Simulation timestep in [s]
                            spawn_position=[0, 0, 0.1],  # Robot position in [m]
                            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
                            drive=4,  # An example of parameter part of the grid search
                            amp_first_last=[0.90, 1.5],  # Set the amplitude gradient
                            phase_bias_body_up = 1.1,
                            phase_bias_body_down = -1.1,
                            body_weights = 0, # Decoupled segments
                            gain_sensory = -2, # Sensory feedback
                            muted_sensors = mix_disrpt[0],
                            muted_oscillators  = mix_disrpt[1],
                            remove_couplings = mix_disrpt[2],
                            seed_disruption = seed_
                        )


                # Save the simulation datas
                os.makedirs('./logs/8g_decoupled/', exist_ok=True)
                filename = './logs/8g_decoupled/simulation_{}_seed{}.{}'
                sim, data = simulation(
                    sim_parameters=sim_parameters,  # Simulation parameters, see above
                    arena='water',  # Can also be 'water', give it a try!
                    fast=True,  # For fast mode (not real-time)
                    headless=True,  # For headless mode (No GUI, could be faster)
                )
                # Log robot data
                data.to_file(filename.format(nb_disrpt, seed_  ,type_disrpt), sim.iteration)
                # Log simulation parameters
                with open(filename.format(nb_disrpt,seed_  , 'pickle'), 'wb') as param_file:
                    pickle.dump(sim_parameters, param_file)



def exercise_8g3(timestep):
    """Exercise 8g3: Disruptions on Combined network (CPG and sensory)"""

    seed = np.arange(1, 11, dtype=int)

    for seed_ in seed:
        for type_disrpt in ['sensors', 'oscillators', 'couplings', 'mixed']:
            if (type_disrpt=='couplings'): ndisrpt=8 
            else: ndisrpt=9 
            for nb_disrpt in range(ndisrpt): 

                if(type_disrpt=='sensors'):
                    # DISRUPTION: MUTED SENSORS
                    sim_parameters = SimulationParameters(
                            duration=30,  # Simulation duration in [s]
                            timestep=timestep,  # Simulation timestep in [s]
                            spawn_position=[0, 0, 0.1],  # Robot position in [m]
                            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
                            drive=4,  # An example of parameter part of the grid search
                            amp_first_last=[0.90, 1.5],  # Set the amplitude gradient
                            phase_bias_body_up = 1.1,
                            phase_bias_body_down = -1.1,
                            gain_sensory = -2, # Sensory feedback
                            muted_sensors = nb_disrpt,
                            seed_disruption = seed_
                        )
                
                if(type_disrpt=='oscillators'):
                    # DISRUPTION: MUTED OSCILLATORS
                    sim_parameters = SimulationParameters(
                            duration=30,  # Simulation duration in [s]
                            timestep=timestep,  # Simulation timestep in [s]
                            spawn_position=[0, 0, 0.1],  # Robot position in [m]
                            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
                            drive=4,  # An example of parameter part of the grid search
                            amp_first_last=[0.90, 1.5],  # Set the amplitude gradient
                            phase_bias_body_up = 1.1,
                            phase_bias_body_down = -1.1,
                            gain_sensory = -2, # Sensory feedback
                            muted_oscillators  = nb_disrpt,
                            seed_disruption = seed_
                        )
                
                if(type_disrpt=='couplings'):
                    # DISRUPTION: MUTED COUPLINGS
                    sim_parameters = SimulationParameters(
                            duration=30,  # Simulation duration in [s]
                            timestep=timestep,  # Simulation timestep in [s]
                            spawn_position=[0, 0, 0.1],  # Robot position in [m]
                            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
                            drive=4,  # An example of parameter part of the grid search
                            amp_first_last=[0.90, 1.5],  # Set the amplitude gradient
                            phase_bias_body_up = 1.1,
                            phase_bias_body_down = -1.1,
                            gain_sensory = -2, # Sensory feedback
                            remove_couplings = nb_disrpt,
                            seed_disruption = seed_
                        )

                if(type_disrpt=='mixed'):
                    # DISRUPTION: MIXED DISRUPTIONS

                    mix_disrpt= [0,0,0]
                    random.seed(6)
                    mix_disrpt[0] = int(np.squeeze(np.random.randint(0,(nb_disrpt+1),1)))
                    mix_disrpt[1] = int(np.squeeze(np.random.randint(0,(nb_disrpt+1-mix_disrpt[0]),1)))
                    mix_disrpt[2] = nb_disrpt-mix_disrpt[0]-mix_disrpt[1]
                    if mix_disrpt[2]==8: mix_disrpt[2]=7
                    np.squeeze(mix_disrpt)
                    print(mix_disrpt)

                    sim_parameters = SimulationParameters(
                            duration=30,  # Simulation duration in [s]
                            timestep=timestep,  # Simulation timestep in [s]
                            spawn_position=[0, 0, 0.1],  # Robot position in [m]
                            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
                            drive=4,  # An example of parameter part of the grid search
                            amp_first_last=[0.90, 1.5],  # Set the amplitude gradient
                            #phase_bias_body_up = 1.1,
                            #phase_bias_body_down = -1.1,
                            gain_sensory = -2, # Sensory feedback
                            muted_sensors = mix_disrpt[0],
                            muted_oscillators  = mix_disrpt[1],
                            remove_couplings = mix_disrpt[2],
                            seed_disruption = seed_
                        )


                # Save the simulation datas
                os.makedirs('./logs/8g_combined/', exist_ok=True)
                filename = './logs/8g_combined/simulation_{}_seed{}.{}'
                sim, data = simulation(
                    sim_parameters=sim_parameters,  # Simulation parameters, see above
                    arena='water',  # Can also be 'water', give it a try!
                    fast=True,  # For fast mode (not real-time)
                    headless=True,  # For headless mode (No GUI, could be faster)
                )
                # Log robot data
                data.to_file(filename.format(nb_disrpt, seed_ , type_disrpt), sim.iteration)
                # Log simulation parameters
                with open(filename.format(nb_disrpt, seed_  ,'pickle'), 'wb') as param_file:
                    pickle.dump(sim_parameters, param_file)





def plot_swim_perf_8g(timestep, network_type='CPG'):
    """Exercise 8g: PLOT SWIMMING PERFORMANCE WITH DISRUPTIONS"""

    # PLOT THE RESULTS
    # Plot effect on swimming performance
    seed = np.arange(1, 11, dtype=int)
    print(seed)

    for type_disrpt in ['sensors', 'oscillators', 'couplings','mixed']:
        if (type_disrpt=='couplings'): ndisrpt=8 
        else: ndisrpt=9 


        energies = np.zeros(ndisrpt) # Compute enery w/ integral
        mean_speed = np.zeros(ndisrpt) # Compute mean_speed
        COT =  np.zeros(ndisrpt) # Cost of transport
        std_energies = np.zeros(ndisrpt)
        std_mean_speed = np.zeros(ndisrpt)
        std_COT = np.zeros(ndisrpt)


        for i in range(ndisrpt):

            energies_2mean = np.zeros_like(seed)
            print(np.shape(energies_2mean))
            mean_speed_2mean = np.zeros(len(seed))
            COT_2mean =  np.zeros(len(seed))

            for idx_seed, seed_ in enumerate(seed):
                # Load data
                data = SalamandraData.from_file(f'logs/8g_{network_type}/simulation_{i}_seed{seed_}.{type_disrpt}')
                with open(f'logs/8g_{network_type}/simulation_{i}_seed{seed_}.pickle', 'rb') as param_file:
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
                

                # Need to discard the initial transient : remove first 3s:
                idx_slice = int(3/timestep)
                head_positions = head_positions[idx_slice:]
                joints_velocities = joints_velocities[idx_slice:]
                joints_torques = joints_torques[idx_slice:]
                times = times[idx_slice:]

                #Compute the energy with sum 
                energies_2mean[idx_seed]= np.sum(np.multiply(joints_velocities, joints_torques)*timestep)


                #Compute the energy with integral 
                for j in range(12): #np.shape(joints_velocities)[1]
                    values2int = np.multiply(joints_velocities[:,j],joints_torques[:,j])
                    energies_2mean[idx_seed] += integrate.simps(values2int,times)

                #Compute speed as time take from starting point to end point
                head_pos_np = np.asarray(head_positions)
                mean_speed_2mean[idx_seed]= np.linalg.norm(head_pos_np[-1]-head_pos_np[0])/(times[-1]-times[0])

                #Compute cost of transport
                COT_2mean[idx_seed]=energies_2mean[idx_seed]/np.linalg.norm(head_pos_np[-1]-head_pos_np[0])

            #Compute the mean for the simulation of a certain seed
            energies[i]= np.mean(energies_2mean)
            std_energies[i]= np.std(energies_2mean)
            mean_speed[i]= np.mean(mean_speed_2mean)
            std_mean_speed[i]= np.std(mean_speed_2mean)
            COT[i]= np.mean(COT_2mean)
            std_COT[i]= np.std(COT_2mean)



        disrpt = np.arange(ndisrpt)
        fig, ax = plt.subplots(3)
        ax[0].errorbar(disrpt, mean_speed, yerr=std_mean_speed, fmt='r-', lw=2)
        ax[0].set_xlabel("Number of neural disruptions")
        ax[0].set_ylabel("Speed [m/s]")
        ax[0].set_ylim((-0.1, 0.7))
        ax[0].grid()
        ax[1].errorbar(disrpt, energies, yerr=std_energies, fmt='b-', lw=2)
        ax[1].set_xlabel("Number of neural disruptions")
        ax[1].set_ylabel("Energy [J]")
        ax[1].grid()
        ax[2].errorbar(disrpt, COT, yerr=std_COT, fmt='g-', lw=2)
        ax[2].set_xlabel("Number of neural disruptions")
        ax[2].set_ylabel("Cost of Transport [ ]")
        ax[2].grid()

        plt.suptitle( f'Neural disruptions on {network_type} network: {type_disrpt} disruption.')
        plt.show()
            


def plot_speed_all_8g(timestep):
    """Exercise 8g: PLOT SPEED FOR ALL NETWORKS AND ALL DISRUPTIONS"""

    # PLOT THE RESULTS

    seed = np.arange(1, 11, dtype=int)

    fig, ax = plt.subplots(4,3)

    # Plot effect on swimming performance
    for idx_network, network_type in enumerate(['CPG', 'decoupled', 'combined']):
        for idx_disrpt, type_disrpt in enumerate(['sensors', 'oscillators', 'couplings','mixed']):
            if (type_disrpt=='couplings'): ndisrpt=8 
            else: ndisrpt=9 

            mean_speed = np.zeros(ndisrpt) # Compute mean_speed
            std_mean_speed = np.zeros(ndisrpt)

            for i in range(ndisrpt):

                mean_speed_2mean = np.zeros(len(seed))

                for idx_seed, seed_ in enumerate(seed):
                    # Load data
                    data = SalamandraData.from_file(f'logs/8g_{network_type}/simulation_{i}_seed{seed_}.{type_disrpt}')
                    with open(f'logs/8g_{network_type}/simulation_{i}_seed{seed_}.pickle', 'rb') as param_file:
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
                    

                    # Need to discard the initial transient : remove first 3s:
                    idx_slice = int(3/timestep)
                    head_positions = head_positions[idx_slice:]
                    times = times[idx_slice:]

                    #Compute speed as time take from starting point to end point
                    head_pos_np = np.asarray(head_positions)
                    mean_speed_2mean[idx_seed]= np.linalg.norm(head_pos_np[-1]-head_pos_np[0])/(times[-1]-times[0])

                mean_speed[i]=np.mean(mean_speed_2mean)
                std_mean_speed[i]= np.std(mean_speed_2mean)

            disrpt = np.arange(ndisrpt)
            
            ax[idx_disrpt][idx_network].errorbar(disrpt, mean_speed, yerr=std_mean_speed, fmt='r-', lw=2)
            ax[idx_disrpt][idx_network].set_ylim((-0.1, 0.7))
            ax[idx_disrpt][idx_network].grid()
            ax[idx_disrpt][idx_network].set_title(f"{network_type}, {type_disrpt}")
            if(idx_network==0):
                ax[idx_disrpt][idx_network].set_ylabel("Speed [m/s]")
            if (idx_disrpt==3):
                ax[idx_disrpt][idx_network].set_xlabel("Number of neural disruptions [ ]")
            
    plt.suptitle('Effect of neural disruptions on speed for different networks.')
    plt.show()
                



if __name__ == '__main__':

    exercise_8g1(timestep=1e-2)
    plot_swim_perf_8g(timestep=1e-2, network_type='CPG') # 'decoupled' , 'combined'
    exercise_8g2(timestep=1e-2)
    plot_swim_perf_8g(timestep=1e-2, network_type='decoupled')
    exercise_8g3(timestep=1e-2)
    plot_swim_perf_8g(timestep=1e-2, network_type='combined')
    plot_speed_all_8g(timestep=1e-2)


