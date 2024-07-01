"""Run network without MuJoCo"""

import time
import numpy as np
import matplotlib.pyplot as plt
from farms_core import pylog
from salamandra_simulation.data import SalamandraState
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
from simulation_parameters import SimulationParameters
from network import SalamandraNetwork


def run_network(duration, update=False, drive=0):
    """Run network without MuJoCo and plot results
    Parameters
    ----------
    duration: <float>
        Duration in [s] for which the network should be run
    update: <bool>
        description
    drive: <float/array>
        Central drive to the oscillators
    """
    # Simulation setup
    timestep = 1e-2
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)
    sim_parameters = SimulationParameters(
        drive=drive,
        amplitude_gradient=None,
        phase_lag=None,
    )
    state = SalamandraState.salamandra_robotica_2(n_iterations)
    network = SalamandraNetwork(sim_parameters, n_iterations, state)
    osc_left = np.arange(8)
    osc_right = np.arange(8, 16)
    osc_legs = np.arange(16, 20)

    # Logs
    phases_log = np.zeros([
        n_iterations,
        len(network.state.phases(iteration=0))
    ])
    phases_log[0, :] = network.state.phases(iteration=0)
    amplitudes_log = np.zeros([
        n_iterations,
        len(network.state.amplitudes(iteration=0))
    ])
    amplitudes_log[0, :] = network.state.amplitudes(iteration=0)
    freqs_log = np.zeros([
        n_iterations,
        len(network.robot_parameters.freqs)
    ])
    freqs_log[0, :] = network.robot_parameters.freqs
    outputs_log = np.zeros([
        n_iterations,
        len(network.get_motor_position_output(iteration=0))
    ])
    outputs_log[0, :] = network.get_motor_position_output(iteration=0)

    outputs_logged = np.zeros([
        n_iterations,
        len(network.outputs(iteration=0))
    ])
    outputs_logged[0,:] = network.outputs(iteration=0)

    nomAmp_logged = np.zeros([
        n_iterations,
        network.robot_parameters.n_oscillators
    ])
    nomAmp_logged[0,:] = network.robot_parameters.nominal_amplitudes

    # Run network ODE and log data
    tic = time.time()
    sim_param = SimulationParameters(duration=duration, timestep=timestep) #defining simulation parameters object for whole simulation
    for i, time0 in enumerate(times[1:]):
        if update:
            sim_param.update_drive(i) #updates the drive (to grow linearly)
            network.robot_parameters.update(
                sim_param
            )
        network.step(i, time0, timestep)
        phases_log[i+1, :] = network.state.phases(iteration=i+1)
        amplitudes_log[i+1, :] = network.state.amplitudes(iteration=i+1)
        outputs_log[i+1, :] = network.get_motor_position_output(iteration=i+1)
        freqs_log[i+1, :] = network.robot_parameters.freqs

        outputs_logged[i+1,:] = network.outputs(iteration=i+1)
        nomAmp_logged[i+1,:] = network.robot_parameters.nominal_amplitudes
    # # Alternative option
    # phases_log[:, :] = network.state.phases()
    # amplitudes_log[:, :] = network.state.amplitudes()
    # outputs_log[:, :] = network.get_motor_position_output()
    toc = time.time()

    # Network performance
    pylog.info('Time to run simulation for {} steps: {} [s]'.format(
        n_iterations,
        toc - tic
    ))

    
    # Implement plots of network results

    ######### GRAPH 1 #########
    fig,ax = plt.subplots(4,1)

    #frequencies
    '''ax[0].plot(times,fr[0][:], label="body")
    ax[0].plot(times,fr[19][:], label="limbs")
    ax[0].set_ylabel("Frequencies [Hz]")
    ax[0].legend(loc='right')'''
    fr = np.multiply(np.diff(phases_log, axis=0), 1/timestep)
    fr = np.transpose(fr)
    for i in range(network.robot_parameters.n_oscillators_body):
        ax[0].plot(times[1:],fr[i][:], color='black')

    for i in range(network.robot_parameters.n_oscillators_body,network.robot_parameters.n_oscillators):
        ax[0].plot(times[1:],fr[i][:], color='grey')
    #ax[0].plot(times[1:],fr[19][:], label="limbs", color='grey')
    ax[0].set_ylabel("Frequencies [Hz]")
    ax[0].legend(loc='right')

    #oscillators
    for i in range(0,4):
        ax[1].plot(times,np.add(outputs_logged[:,i], np.pi*i/3), color='tab:blue')
        ax[1].text(duration-1,outputs_logged[int((duration-1)/timestep),i]+np.pi*i/3+0.2, "x{}".format(i+1), horizontalalignment='center',size='x-small')
    for i in range(4,8):
        ax[1].plot(times,np.add(outputs_logged[:,i], np.pi*i/3), color='tab:orange')
        ax[1].text(duration-1,outputs_logged[int((duration-1)/timestep),i]+np.pi*i/3+0.2, "x{}".format(i+1), horizontalalignment='center',size='x-small')

   
    ax[1].set_ylabel("x body")
    ax[1].set_yticks([])

    ax[2].plot(times,outputs_logged[:,16])
    ax[2].text(duration-1,outputs_logged[int((duration-1)/timestep),17]+0.1, "x17", horizontalalignment='center',size='x-small')
    ax[2].plot(times,np.add(outputs_logged[:,18], np.pi/3))
    ax[2].text(duration-1,outputs_logged[int((duration-1)/timestep),17]+0.1+np.pi/3, "x19", horizontalalignment='center',size='x-small')
    
    ####
    ax[2].plot(times,np.add(outputs_logged[:,17], 3*np.pi/3))
    ax[2].text(duration-1,outputs_logged[int((duration-1)/timestep),17]+0.1+3*np.pi/3, "x18", horizontalalignment='center',size='x-small')
    ax[2].plot(times,np.add(outputs_logged[:,19], 2*np.pi/3))
    ax[2].text(duration-1,outputs_logged[int((duration-1)/timestep),17]+0.1+2*np.pi/3, "x20", horizontalalignment='center',size='x-small')
    ####

    ax[2].set_ylabel("x limb")
    ax[2].set_yticks([])

    ax[3].plot(times,sim_param.drive_array, color='black')
    ax[3].set_ylabel("Drive")
    ax[3].set_xlabel("Time [s]")

    for i in range(4):
        ax[i].axvline(x=np.interp(sim_param.d_low_body, sim_param.drive_array, times),color='grey', linestyle='--', linewidth=0.7)
        ax[i].axvline(x=np.interp(sim_param.d_low_limb, sim_param.drive_array, times),color='grey', linestyle='--', linewidth=0.7)
        ax[i].axvline(x=np.interp(sim_param.d_high_body, sim_param.drive_array, times),color='grey', linestyle='--', linewidth=0.7)
        ax[i].axvline(x=np.interp(sim_param.d_high_limb, sim_param.drive_array, times),color='grey', linestyle='--', linewidth=0.7)

    plt.grid(axis='y')
    plt.legend()
    plt.show()
       
    ######### GRAPH 2 #########
    fig,ax=plt.subplots(2,1)
    fr = np.transpose(freqs_log)
    ax[0].plot(sim_param.drive_array, fr[0][:], label="body")
    ax[0].plot(sim_param.drive_array, fr[19][:], label="limbs")
    ax[0].set_ylabel("Frequencies [Hz]")
    ax[0].legend(loc='upper left')

    ax[1].plot(sim_param.drive_array, nomAmp_logged[:,0], label="body")
    ax[1].plot(sim_param.drive_array, nomAmp_logged[:,19], label="limbs")
    ax[1].set_ylabel("Nominal amplitudes \n R [rad]")
    ax[1].legend(loc='upper left')
    plt.show()


    ######### GRAPH 3 #########
    fig,ax = plt.subplots(4,1)

    ax[0].plot(times,np.add(outputs_logged[:,0], np.pi/3))
    ax[0].text(duration-1,outputs_logged[int((duration-1)/timestep),0]+np.pi/3+0.2, "x1", horizontalalignment='center',size='x-small')
    ax[0].plot(times,outputs_logged[:,17])
    ax[0].text(duration-1,outputs_logged[int((duration-1)/timestep),17]+0.2, "x18", horizontalalignment='center',size='x-small')
    ax[0].set_ylabel("x")
    ax[0].set_yticks([])

    ax[1].plot(times,fr[0][:], label="body")
    ax[1].plot(times,fr[19][:], label="limbs")
    ax[1].set_ylabel("Frequencies [Hz]")
    ax[1].legend(loc='upper left')

    ax[2].plot(times, amplitudes_log[:,0], label="body")
    ax[2].plot(times, amplitudes_log[:,19], label="limbs")
    ax[2].set_ylabel("Amplitudes [rad]")
    ax[2].legend(loc='upper left')

    ax[3].plot(times,sim_param.drive_array, color='black')
    ax[3].set_ylabel("Drive")
    ax[3].set_xlabel("Time [s]")

    for i in range(4):
        ax[i].axvline(x=np.interp(sim_param.d_low_body, sim_param.drive_array, times),color='grey', linestyle='--', linewidth=0.7)
        ax[i].axvline(x=np.interp(sim_param.d_low_limb, sim_param.drive_array, times),color='grey', linestyle='--', linewidth=0.7)
        ax[i].axvline(x=np.interp(sim_param.d_high_body, sim_param.drive_array, times),color='grey', linestyle='--', linewidth=0.7)
        ax[i].axvline(x=np.interp(sim_param.d_high_limb, sim_param.drive_array, times),color='grey', linestyle='--', linewidth=0.7)

    plt.grid(axis='y')
    plt.legend()
    plt.show()


def main(plot):
    """Main"""

    run_network(duration=40, update=True)
    
    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=not save_plots())

