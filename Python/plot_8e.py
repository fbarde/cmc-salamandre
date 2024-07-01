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
    sim_param =  SimulationParameters(
            duration=20,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=4,
            body_weights = 0
        )
    state = SalamandraState.salamandra_robotica_2(n_iterations)
    network = SalamandraNetwork(sim_param, n_iterations, state)
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
     #defining simulation parameters object for whole simulation

    for i, time0 in enumerate(times[1:]):
        if update:
            #sim_param.update_drive(i) #updates the drive (to grow linearly)
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
    fig,ax = plt.subplots(1,1)

    #oscillators
    for i in range(0,4):
        ax.plot(times,np.add(outputs_logged[:,i], np.pi*i/3), color='tab:blue')
        ax.text(duration-1,outputs_logged[int((duration-1)/timestep),i]+np.pi*i/3+0.2, "x{}".format(i+1), horizontalalignment='center',size='x-small')
    for i in range(4,8):
        ax.plot(times,np.add(outputs_logged[:,i], np.pi*i/3), color='tab:orange')
        ax.text(duration-1,outputs_logged[int((duration-1)/timestep),i]+np.pi*i/3+0.2, "x{}".format(i+1), horizontalalignment='center',size='x-small')
    for i in range(8,12):
        ax.plot(times,np.add(outputs_logged[:,i], np.pi*i/3), color='tab:blue')
        ax.text(duration-1,outputs_logged[int((duration-1)/timestep),i]+np.pi*i/3+0.2, "x{}".format(i+1), horizontalalignment='center',size='x-small')
    for i in range(12,16):
        ax.plot(times,np.add(outputs_logged[:,i], np.pi*i/3), color='tab:orange')
        ax.text(duration-1,outputs_logged[int((duration-1)/timestep),i]+np.pi*i/3+0.2, "x{}".format(i+1), horizontalalignment='center',size='x-small')

   
    ax.set_ylabel("x body")
    ax.set_yticks([])
    
    plt.grid(axis='y')
    plt.legend()
    plt.show()
       

def main(plot):
    """Main"""

    run_network(duration=20, update=True)
    
    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=not save_plots())

