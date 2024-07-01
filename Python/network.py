"""Oscillator network ODE"""

import numpy as np
from scipy.integrate import ode
from robot_parameters import RobotParameters


def network_ode(_time, state, robot_parameters,loads=None):
    """Network_ODE

    Parameters
    ----------
    _time: <float>
        Time
    state: <np.array>
        ODE states at time _time
    robot_parameters: <RobotParameters>
        Instance of RobotParameters
    loads: <np.array>
        The lateral forces applied to the body links

    Returns
    -------
    :<np.array>
        Returns derivative of state (phases and amplitudes)

    """
    n_oscillators = robot_parameters.n_oscillators
    phases = state[:n_oscillators]
    amplitudes = state[n_oscillators:2*n_oscillators]
    # Implement equation here
    f = robot_parameters.freqs 
    w = robot_parameters.coupling_weights
    R = robot_parameters.nominal_amplitudes
    a = robot_parameters.rates
    phi = robot_parameters.phase_bias

    dtheta = np.zeros_like(phases)
    dr = np.zeros_like(amplitudes)


    for i in range(n_oscillators):
        dtheta[i] = 2*np.pi*f[i]
        for j in range(n_oscillators):
            dtheta[i] += amplitudes[j]*w[i][j]*np.sin(phases[j]-phases[i]-phi[i][j])

        dr[i] = a[i]*(R[i]-amplitudes[i])


    return np.concatenate([dtheta, dr])

# Function used starting 8.e with sensory feedback
def network_ode_feedback(_time, state, robot_parameters,loads):
    """Network_ODE_Feedback

    Parameters
    ----------
    _time: <float>
        Time
    state: <np.array>
        ODE states at time _time
    robot_parameters: <RobotParameters>
        Instance of RobotParameters
    loads: <np.array>
        The lateral forces applied to the body links

    Returns
    -------
    :<np.array>
        Returns derivative of state (phases and amplitudes)

    """
    n_oscillators = robot_parameters.n_oscillators
    phases = state[:n_oscillators]
    amplitudes = state[n_oscillators:2*n_oscillators]
    # Implement equation here
    f = robot_parameters.freqs 
    w = robot_parameters.coupling_weights
    R = robot_parameters.nominal_amplitudes
    a = robot_parameters.rates
    phi = robot_parameters.phase_bias
    dtheta = np.zeros_like(phases)
    dr = np.zeros_like(amplitudes)

    # Gain w_fb for sensory feedback
    gain = robot_parameters.gain
    
    for i in range(n_oscillators):
        if i < 8 :
            dtheta[i] = 2*np.pi*f[i] + gain[i]*np.cos(phases[i])*(loads[i])
            dtheta[i+8] = 2*np.pi*f[i] + gain[i]*np.cos(phases[i])*loads[i] #[1]
        if i>15 : 
            dtheta[i] = 2*np.pi*f[i]
        for j in range(n_oscillators):
            dtheta[i] += amplitudes[j]*w[i][j]*np.sin(phases[j]-phases[i]-phi[i][j])

        dr[i] = a[i]*(R[i]-amplitudes[i])


    return np.concatenate([dtheta, dr])


def motor_output(phases, amplitudes, iteration):
    """Motor output

    Parameters
    ----------
    phases: <np.array>
        Phases of the oscillator
    amplitudes: <np.array>
        Amplitudes of the oscillator

    Returns
    -------
    : <np.array>
        Motor outputs for joint in the system.

    """
    q = np.zeros_like(phases)[:12]
    for i in range(0,8):
        if amplitudes[17]>0:
            q[i] = 0.7*(amplitudes[i]*(1+np.cos(phases[i])) - amplitudes[i+8]*(1+np.cos(phases[i+8])))
        else:
            q[i] = (amplitudes[i]*(1+np.cos(phases[i])) - amplitudes[i+8]*(1+np.cos(phases[i+8])))
            
    for j in range(8,12):
        q[j] = phases[j+8] #-np.pi/2 #amplitudes[j+8]*(1+np.cos(phases[j+8]))
        
    return q


class SalamandraNetwork:
    """Salamandra oscillator network"""

    def __init__(self, sim_parameters, n_iterations, state,initial_phase=1e-4):
        super().__init__()
        self.n_iterations = n_iterations
        # States
        self.state = state
        # Parameters
        self.robot_parameters = RobotParameters(sim_parameters)

        # Set initial state
        # Replace your oscillator phases here
        if abs(initial_phase-1e-4)<1e-6 :     
            self.state.set_phases(
                iteration=0,
                value=initial_phase*np.random.ranf(self.robot_parameters.n_oscillators),
            )
        else : 
            self.state.set_phases_left(iteration=0,value = initial_phase* np.arange(1,self.robot_parameters.n_body_joints+1,dtype = float)[::-1])
            self.state.set_phases_right(iteration=0,value = initial_phase*np.arange(self.robot_parameters.n_body_joints+1,2*(self.robot_parameters.n_body_joints)+1,dtype = float)[::-1])
            self.state.set_phases_legs(iteration=0,value = initial_phase*np.random.ranf(self.robot_parameters.n_oscillators_legs))

        self.solver = ode(f=network_ode_feedback)

        self.solver.set_integrator('dopri5')
        self.solver.set_initial_value(y=self.state.array[0], t=0.0)

    def step(self, iteration, time, timestep, loads=None):
        """Step"""
        
        if loads is None:
            loads = np.zeros(self.robot_parameters.n_joints)
        if iteration + 1 >= self.n_iterations:
            return
        self.solver.set_f_params(self.robot_parameters,loads)
        self.state.array[iteration+1, :] = self.solver.integrate(time+timestep)

    def outputs(self, iteration=None):
        """Oscillator outputs"""
        x = np.zeros(self.robot_parameters.n_oscillators)
        for i in range(0,self.robot_parameters.n_oscillators_body):
            x[i] = self.state.amplitudes(iteration=iteration)[i]*(1+np.cos(self.state.phases(iteration=iteration)[i]))

        for j in range(self.robot_parameters.n_oscillators_body,self.robot_parameters.n_oscillators):
            x[j] = self.state.amplitudes(iteration=iteration)[j]*(1+np.cos(self.state.phases(iteration=iteration)[j]))
        return x

    def get_motor_position_output(self, iteration=None):
        """Get motor position"""
        return motor_output(
            self.state.phases(iteration=iteration),
            self.state.amplitudes(iteration=iteration),
            iteration=iteration,
        )

