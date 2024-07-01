"""Exercise 9a"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
from network import SalamandraNetwork
from salamandra_simulation.data import SalamandraData
import matplotlib.pyplot as plt
from plot_results import plot_positions, plot_trajectory


def exercise_9a1(timestep):
    """Exercise 9a"""
    # Parameters

    parameter_set = [
        SimulationParameters(
            duration=12,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=2.7,  # An example of parameter part of the grid search
            phase_bias_limb_body = -np.pi/2

        )
    ]

    # Grid search
    os.makedirs('./logs/example/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/example/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='ground',
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, '9a1'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
    
    data = SalamandraData.from_file('logs/example/simulation_0.9a1')
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
    fig, ax = plt.subplots(2,1,sharex=True,sharey=True)
    for i in range(0,4):
        ax[0].plot(times,np.add(x[:,i], np.pi*i/3), color='tab:blue')
        ax[0].text(duration-1,x[int((duration-1)/timestep),i]+np.pi*i/3+0.2, "x{}".format(i+1), horizontalalignment='center',size='large')
    for i in range(4,8):
        ax[0].plot(times,np.add(x[:,i], np.pi*i/3), color='tab:blue')
        ax[0].text(duration-1,x[int((duration-1)/timestep),i]+np.pi*i/3+0.2, "x{}".format(i+1), horizontalalignment='center',size='large')
    for i in range(8,12):
        ax[1].plot(times,np.add(x[:,i], np.pi*(i-8)/3), color='tab:orange')
        ax[1].text(duration-1,x[int((duration-1)/timestep),i]+np.pi*(i-8)/3+0.2, "x{}".format(i+1), horizontalalignment='center',size='large')
    for i in range(12,16):
        ax[1].plot(times,np.add(x[:,i], np.pi*(i-8)/3), color='tab:orange')
        ax[1].text(duration-1,x[int((duration-1)/timestep),i]+np.pi*(i-8)/3+0.2, "x{}".format(i+1), horizontalalignment='center',size='large')

    ax[0].plot(times,np.add(x[:,16],9*np.pi/3), color='tab:blue')
    ax[0].text(duration-1,x[int((duration-1)/timestep),17]+0.1+9*np.pi/3, "x17", horizontalalignment='center',size='large')
    ax[0].plot(times,np.add(x[:,18], 10*np.pi/3), color='tab:blue')
    ax[0].text(duration-1,x[int((duration-1)/timestep),17]+0.1+10*np.pi/3, "x19", horizontalalignment='center',size='large')
    
    ####
    ax[1].plot(times,np.add(x[:,17], 9*np.pi/3),color='tab:orange')
    ax[1].text(duration-1,x[int((duration-1)/timestep),17]+0.1+9*np.pi/3, "x18", horizontalalignment='center',size='large')
    ax[1].plot(times,np.add(x[:,19], 10*np.pi/3),color='tab:orange')
    ax[1].text(duration-1,x[int((duration-1)/timestep),17]+0.1+10*np.pi/3, "x20", horizontalalignment='center',size='large')
   
    ax[0].set_title('Oscillator patterns for the oscillators (left side)',size = 14)
    #ax[0].set_xlabel('Time [s]',form1)
    ax[0].set_ylabel("output x",form1)
    ax[0].set_yticks([])

    ax[1].set_title('Oscillator patterns for the oscillators (right side)',size = 14)
    #ax[1].set_xlabel('Time [s]',form1)
    ax[1].set_ylabel("output x",form1)
    ax[1].set_yticks([])
    
    plt.grid(axis='y')
    plt.xlabel('Time [s]',form1)
    plt.legend()
    plt.show()

    links_positions = data.sensors.links.urdf_positions()
    head_positions = links_positions[:, 0, :]
    tail_positions = links_positions[:, 8, :]
    #if you want the value as numpy array do : np.array(link_position[:, 10, :])
    joints_velocities = data.sensors.joints.velocities_all()
    joints_torques = data.sensors.joints.motor_torques_all()
    joints_positions = data.sensors.joints.positions_all()

    


    plt.figure()
    plot_trajectory(head_positions)

    plt.figure()
    plot_positions(times,head_positions)
    plt.show()

    # Need to discard the initial transient : remove first 3s:
    idx_slice = int(3.4/timestep)
    links_positions = links_positions[idx_slice:]
    times = times[idx_slice:]

    # Get the limb positions
    limb_positions1 = links_positions[:, 10, :]
    limb_positions2 = links_positions[:, 11, :]
    limb_positions3 = links_positions[:, 12, :]
    limb_positions4 = links_positions[:, 13, :]
    limb_positions5 = links_positions[:, 14, :]
    limb_positions6 = links_positions[:, 15, :]
    limb_positions7 = links_positions[:, 16, :]
    limb_positions8 = links_positions[:, 17, :]
    #limb_positions9 = links_positions[:, 18, :]

    print(limb_positions1[0])

    num_idx = int(170e-3/timestep)
    num_plots=8
    pos_links = np.zeros((num_plots,9,2)) #Shape : 9 plots x 9 links x 2 coord (x and y)

    fig, ax = plt.subplots(nrows=8, figsize=(6, 25))
    for i in range(num_plots):

        pos_links[i]= links_positions[i*num_idx, 0:9, 0:2]
        time = format(times[i*num_idx] , '.2f')
        ax[i].plot(pos_links[i,:,0], pos_links[i,:,1], 's-', lw=2, label=f'{time} s')
        ax[i].plot(limb_positions1[i*num_idx,0],limb_positions1[i*num_idx,1], 'sm', lw=1.5)
        ax[i].plot(limb_positions2[i*num_idx,0],limb_positions2[i*num_idx,1], 'sm', lw=1.5)
        ax[i].plot(limb_positions3[i*num_idx,0],limb_positions3[i*num_idx,1], 'sm', lw=1.5)
        ax[i].plot(limb_positions4[i*num_idx,0],limb_positions4[i*num_idx,1], 'sm', lw=1.5)
        ax[i].plot(limb_positions5[i*num_idx,0],limb_positions5[i*num_idx,1], 'sm', lw=1.5)
        ax[i].plot(limb_positions6[i*num_idx,0],limb_positions6[i*num_idx,1], 'sm', lw=1.5)
        ax[i].plot(limb_positions7[i*num_idx,0],limb_positions7[i*num_idx,1], 'sm', lw=1.5)
        ax[i].plot(limb_positions8[i*num_idx,0],limb_positions8[i*num_idx,1], 'sm', lw=1.5)
        #ax[i].plot(limb_positions9[i*num_idx,0],limb_positions9[i*num_idx,1], 'sm', lw=1.5)
        
        #Plot the segments between the legs:
        ax[i].plot([limb_positions1[i*num_idx,0],limb_positions2[i*num_idx,0]],[limb_positions1[i*num_idx,1],limb_positions2[i*num_idx,1]], '-m', lw=1.5)
        ax[i].plot([limb_positions1[i*num_idx,0],limb_positions3[i*num_idx,0]],[limb_positions1[i*num_idx,1],limb_positions3[i*num_idx,1]], '-m', lw=1.5)
        ax[i].plot([limb_positions3[i*num_idx,0],limb_positions4[i*num_idx,0]],[limb_positions3[i*num_idx,1],limb_positions4[i*num_idx,1]], '-m', lw=1.5)
        ax[i].plot([limb_positions5[i*num_idx,0],limb_positions6[i*num_idx,0]],[limb_positions5[i*num_idx,1],limb_positions6[i*num_idx,1]], '-m', lw=1.5)
        ax[i].plot([limb_positions7[i*num_idx,0],limb_positions8[i*num_idx,0]],[limb_positions7[i*num_idx,1],limb_positions8[i*num_idx,1]], '-m', lw=1.5)
        ax[i].plot([limb_positions5[i*num_idx,0],limb_positions7[i*num_idx,0]],[limb_positions5[i*num_idx,1],limb_positions7[i*num_idx,1]], '-m', lw=1.5)


        #ax[i].vlines(x=pos_links[0,0,0], ymin= min(pos_links[i,:,1])-0.01, ymax=max(pos_links[i,:,1])+0.01, color='red', linestyles='dashed')
        ax[i].grid()
            
        ax[i].legend(loc='upper right')
    ax[7].set_xlabel('x [m]')
    ax[4].set_ylabel('y [m]')
    plt.suptitle('Body deformations (links positions) of the salameter at different times.')
    plt.show()

    
    pass




def exercise_9a3(timestep,several_phase=False):
    """Exercise 9a"""
    # Parameters
    phase_bias = np.linspace(0,np.pi*2,25)
    if several_phase==True:
        parameter_set=[]
        for phase in phase_bias : 
            sim_param = [SimulationParameters(
            duration=15,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=2.7, 
            phase_bias_limb_body = phase
            )]

            parameter_set = np.concatenate((parameter_set,sim_param))

    # Grid search on the phase bias between limb and body
        os.makedirs('./logs/example/', exist_ok=True)
        for simulation_i, sim_parameters in enumerate(parameter_set):
            filename = './logs/example/simulation_{}.{}'
            sim, data = simulation(
                sim_parameters=sim_parameters,  # Simulation parameters, see above
                arena='ground',
                # Can also be 'water', give it a try!
                fast=True,  # For fast mode (not real-time)
                headless=True,  # For headless mode (No GUI, could be faster)
            )
            # Log robot data
            data.to_file(filename.format(simulation_i, '9a3'), sim.iteration)
            # Log simulation parameters
            with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
                pickle.dump(sim_parameters, param_file)

    N=len(phase_bias)
    energy = np.zeros(N)
    mean_speed = np.zeros(N)
    COT = np.zeros(N)

    for i in range(N):
        # Load data
        data = SalamandraData.from_file(f'logs/example/simulation_{i}.9a3')
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
        mean_speed[i]= np.linalg.norm(head_pos_np[-1]-head_pos_np[0])/(times[-1]-times[0])

        #try with cost of transport. COT = E/(mgd)
        COT[i] =  energy[i]/np.linalg.norm(head_pos_np[-1]-head_pos_np[0])

    form1 = {'size': 14}

    plt.figure()
    plt.scatter(phase_bias,energy,marker ="+")
    plt.xlabel("Phase offset [rad]",fontdict=form1)
    plt.ylabel("Energy [J]",fontdict=form1)
    plt.title("Energy depending on phase offset",fontdict=form1)
    plt.grid()
    plt.show()

    plt.figure()
    plt.scatter(phase_bias,mean_speed,marker ="+")
    plt.xlabel("Phase offset [rad]",fontdict=form1)
    plt.ylabel("Speed [m/s]",fontdict=form1)
    plt.title(" Walking speed depending on phase offset",fontdict=form1)
    plt.grid()
    plt.show()

    plt.figure()
    plt.scatter(phase_bias,COT,marker ="+")
    plt.xlabel("Phase offset [rad]",fontdict=form1)
    plt.ylabel("COT []",fontdict=form1)
    plt.title(" Cost of Transport depending on phase offset",fontdict=form1)
    plt.grid()
    plt.show()

    pass

def exercise_9a4(timestep,several_amps=False):
    """Exercise 9a"""
    # Parameters
    nom_amps = np.arange(0,1.5,0.05)
    if several_amps==True:
        parameter_set=[]
        for amp in nom_amps : 
            sim_param = [SimulationParameters(
            duration=15,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=2.7, 
            phase_bias_limb_body = 5*np.pi/3,
            nom_amp0_body = amp,
            nom_amp_drive_body = 0
            )]

            parameter_set = np.concatenate((parameter_set,sim_param))

    # Grid search on nominal amplitudes of the body oscillators
        os.makedirs('./logs/example/', exist_ok=True)
        for simulation_i, sim_parameters in enumerate(parameter_set):
            filename = './logs/example/simulation_{}.{}'
            sim, data = simulation(
                sim_parameters=sim_parameters,  # Simulation parameters, see above
                arena='ground',
                fast=True,  # For fast mode (not real-time)
                headless=True,  # For headless mode (No GUI, could be faster)
            )
            # Log robot data
            data.to_file(filename.format(simulation_i, '9a4'), sim.iteration)
            # Log simulation parameters
            with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
                pickle.dump(sim_parameters, param_file)

    N=len(nom_amps)
    energy = np.zeros(N)
    mean_speed = np.zeros(N)
    COT = np.zeros(N)

    for i in range(N):
        # Load data
        data = SalamandraData.from_file(f'logs/example/simulation_{i}.9a4')
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
        mean_speed[i]= np.linalg.norm(head_pos_np[-1]-head_pos_np[0])/(times[-1]-times[0])

        #try with cost of transport. COT = E/(mgd)
        COT[i] =  energy[i]/np.linalg.norm(head_pos_np[-1]-head_pos_np[0])

    form1 = {'size': 14}
    print("AMP : \n",nom_amps, "\nSPEED : \n", mean_speed, "\nCOT : \n", COT)

    plt.figure()
    plt.scatter(nom_amps,energy,marker ="+")
    plt.xlabel("Nominal amplitude",fontdict=form1)
    plt.ylabel("Energy [J]",fontdict=form1)
    plt.title("Energy depending on nominal amplitude",fontdict=form1)
    plt.grid()
    plt.show()

    plt.figure()
    plt.scatter(nom_amps,mean_speed,marker ="+")
    plt.xlabel("Nominal amplitude",fontdict=form1)
    plt.ylabel("Speed [m/s]",fontdict=form1)
    plt.title(" Walking speed depending on nominal amplitude",fontdict=form1)
    plt.grid()
    plt.show()

    plt.figure()
    plt.scatter(nom_amps,COT,marker ="+")
    plt.xlabel("Nominal amplitude",fontdict=form1)
    plt.ylabel("COT []",fontdict=form1)
    plt.title(" Cost of Transport depending on nominal amplitude",fontdict=form1)
    plt.grid()
    plt.show()

def exercise_9a4_sim(timestep):
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=20,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=2.7,
            phase_bias_limb_body = 5*np.pi/3,
            nom_amp0_body = 0.2,
            nom_amp_drive_body = 0.1
        )
    ]

    # Grid search
    os.makedirs('./logs/example/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/example/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters, 
            arena='ground'
        )

if __name__ == '__main__':
    exercise_9a1(timestep=1e-2)
    #exercise_9a3(timestep=1e-2,several_phase=True)
    exercise_9a4_sim(timestep=1e-2)
    #exercise_9a4(timestep=1e-2,several_amps=True)

