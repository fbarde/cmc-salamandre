"""Robot parameters"""

import numpy as np
import random
from farms_core import pylog


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super(RobotParameters, self).__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([
            self.n_oscillators,
            self.n_oscillators,
        ])
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)

        self.gain = np.zeros_like(self.n_oscillators)
        
        # parameter to mute sensors (loads=0, wbf=0) : 
        self.muted_sensors = parameters.muted_sensors

        # parameter to remove_couplings (w=0): 
        self.remove_couplings = parameters.remove_couplings

        # parameter to mute oscillators (f=0) : 
        self.muted_oscillators = parameters.muted_oscillators
        self.sim_parameters = parameters

        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # theta_i
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i
        self.set_gain(parameters) # w_fb

    def set_frequencies(self, parameters):
        """Set frequencies"""
        drive_left = parameters.drive/parameters.turn
        drive_right = parameters.drive*parameters.turn
        for i in range(self.n_oscillators_body):
            if i<self.n_body_joints:
                #setting freqs for left part of the body
                if((drive_left >= parameters.d_low_body) and (drive_left <= parameters.d_high_body)):
                    self.freqs[i] = parameters.freq0_body +parameters.freq_drive_body*drive_left
                else:
                    self.freqs[i] = parameters.f_sat_body
            
            if i>=self.n_body_joints:
                #setting freqs for right part of the body
                if((drive_right >= parameters.d_low_body) and (drive_right <= parameters.d_high_body)):
                    self.freqs[self.n_body_joints:self.n_oscillators_body] = parameters.freq0_body +parameters.freq_drive_body*drive_right
                else:
                    self.freqs[self.n_body_joints:self.n_oscillators_body] = parameters.f_sat_body

        #setting freqs for left limbs
        if((drive_left >= parameters.d_low_limb) and (drive_left <= parameters.d_high_limb)):
            self.freqs[self.n_oscillators_body] = parameters.freq0_limb +parameters.freq_drive_limb*drive_left
            self.freqs[self.n_oscillators_body+2] = parameters.freq0_limb +parameters.freq_drive_limb*drive_left
        else:
            self.freqs[self.n_oscillators_body] = parameters.f_sat_limb
            self.freqs[self.n_oscillators_body+2] = parameters.f_sat_limb

        #setting freqs for right limbs
        if((drive_right >= parameters.d_low_limb) and (drive_right <= parameters.d_high_limb)):
            self.freqs[self.n_oscillators_body+1] = parameters.freq0_limb +parameters.freq_drive_limb*drive_right
            self.freqs[self.n_oscillators_body+3] = parameters.freq0_limb +parameters.freq_drive_limb*drive_right
        else:
            self.freqs[self.n_oscillators_body+1] = parameters.f_sat_limb
            self.freqs[self.n_oscillators_body+3] = parameters.f_sat_limb

        #disruption : 
        if parameters.muted_oscillators > 0 : 
            np.random.seed(parameters.seed_disruption+20)  #replicating the results when running the simulation again using the same seed.
            idx = random.sample(range(0,self.n_body_joints),parameters.muted_oscillators)
            #idx = np.random.randint(0,self.n_body_joints,parameters.muted_oscillators)
            for i in idx :
                self.freqs[i] = 0
                self.freqs[i+self.n_body_joints] = 0
        
    def set_coupling_weights(self, parameters):
        #make array of which indexes are what
        """Set coupling weights"""
        for i in range(0,16):
            if(i!=0):
                self.coupling_weights[i][i-1] = parameters.body_weights #10
            if(i!=15):
                self.coupling_weights[i][i+1] = parameters.body_weights
            if(i<8):
                self.coupling_weights[i][i+self.n_body_joints] = parameters.contra_body_weights
            if(i>7):
                self.coupling_weights[i][i-self.n_body_joints] = parameters.contra_body_weights
        
        for j in range(0,4):
            self.coupling_weights[j][16] = parameters.body_limb_weights
            self.coupling_weights[j+8][17] = parameters.body_limb_weights
            self.coupling_weights[j+4][18] = parameters.body_limb_weights
            self.coupling_weights[j+12][19] = parameters.body_limb_weights

        for k in [16,17]:
            self.coupling_weights[k][k+2] = parameters.limb_weights #10
            self.coupling_weights[k+2][k] = parameters.limb_weights

        for l in [16,18]:
            self.coupling_weights[l][l+1] = parameters.limb_weights
            self.coupling_weights[l+1][l] = parameters.limb_weights  

        # Disruptions:
        if parameters.remove_couplings > 0 : 
            random.seed(parameters.seed_disruption+20)  #replicating the results when running the simulation again using the same seed.
            idx = random.sample(range(0,self.n_body_joints-1),parameters.remove_couplings)
            for i in idx :
                #Need to remove 4 coupling weights for each disruption
                #Remove for the two oscillators in a same body joint
                self.coupling_weights[i][i+1] = 0
                self.coupling_weights[i+self.n_body_joints][i+self.n_body_joints+1] = 0
                #Remove up and down for both oscillators
                self.coupling_weights[i+1][i] = 0
                self.coupling_weights[i+self.n_body_joints+1][i+self.n_body_joints] = 0

    def set_phase_bias(self, parameters):
        """Set phase bias"""
        for i in range(0,self.n_oscillators_body):
            if(i!=0):
                self.phase_bias[i][i-1] = parameters.phase_bias_body_up #pi/4
            if(i!=self.n_oscillators_body-1):
                self.phase_bias[i][i+1] = parameters.phase_bias_body_down #-pi/4
            if(i<self.n_body_joints):
                self.phase_bias[i][i+self.n_body_joints] = parameters.phase_bias_body_contralat #pi
            if(i>self.n_body_joints-1):
                self.phase_bias[i][i-self.n_body_joints] = parameters.phase_bias_body_contralat

        for j in range(0,4):
            self.phase_bias[j][16] = parameters.phase_bias_limb_body #pi
            self.phase_bias[j+8][17] = parameters.phase_bias_limb_body
            self.phase_bias[j+4][18] = parameters.phase_bias_limb_body
            self.phase_bias[j+12][19] = parameters.phase_bias_limb_body
        
        for k in [16,17]:
            self.phase_bias[k][k+2] = parameters.phase_bias_limb #pi
            self.phase_bias[k+2][k] = parameters.phase_bias_limb

        for l in [16,18]:
            self.phase_bias[l][l+1] = parameters.phase_bias_limb
            self.phase_bias[l+1][l] = parameters.phase_bias_limb

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        self.rates[:] = parameters.amp_rates

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        gradient_amp = np.linspace(parameters.amp_first_last[0],parameters.amp_first_last[1],self.n_body_joints)
        drive_left = parameters.drive/parameters.turn
        drive_right = parameters.drive*parameters.turn
        for i in range(self.n_oscillators_body):
            if i<self.n_body_joints:
                #setting nom_amp for left part of the body
                if((drive_left >= parameters.d_low_body) and (drive_left <= parameters.d_high_body)):
                    self.nominal_amplitudes[i] =  gradient_amp[i]*(parameters.nom_amp0_body +drive_left*parameters.nom_amp_drive_body)
                else :
                    self.nominal_amplitudes[i] = parameters.nom_amp_sat
            else:
                #setting nom_amp for right part of the body
                if((drive_right >= parameters.d_low_body) and (drive_right <= parameters.d_high_body)):
                    self.nominal_amplitudes[i] =  gradient_amp[i-8]*(parameters.nom_amp0_body +drive_right*parameters.nom_amp_drive_body)
                else :
                    self.nominal_amplitudes[i] = parameters.nom_amp_sat

        #setting nom_amp for left limbs
        if((drive_left >= parameters.d_low_limb) and (drive_left <= parameters.d_high_limb)):
            self.nominal_amplitudes[self.n_oscillators_body] = parameters.nom_amp0_limb +drive_left*parameters.nom_amp_drive_limb
            self.nominal_amplitudes[self.n_oscillators_body+2] = parameters.nom_amp0_limb +drive_left*parameters.nom_amp_drive_limb
        else:
            self.nominal_amplitudes[self.n_oscillators_body] = parameters.nom_amp_sat
            self.nominal_amplitudes[self.n_oscillators_body+2] = parameters.nom_amp_sat
        
        #setting nom_amp for right limbs
        if((drive_right >= parameters.d_low_limb) and (drive_right <= parameters.d_high_limb)):
            self.nominal_amplitudes[self.n_oscillators_body+1] = parameters.nom_amp0_limb +drive_right*parameters.nom_amp_drive_limb
            self.nominal_amplitudes[self.n_oscillators_body+3] = parameters.nom_amp0_limb +drive_right*parameters.nom_amp_drive_limb
        else:
            self.nominal_amplitudes[self.n_oscillators_body+1] = parameters.nom_amp_sat
            self.nominal_amplitudes[self.n_oscillators_body+3] = parameters.nom_amp_sat
    '''
        if((parameters.drive >= parameters.d_low_body) and (parameters.drive <= parameters.d_high_body)):
            for i in range(self.n_oscillators_body) : 
                if i < 8 : 
                    self.nominal_amplitudes[i] = gradient_amp[i]*(parameters.nom_amp0_body +parameters.drive*parameters.nom_amp_drive_body)
                    self.nominal_amplitudes[i+8] = gradient_amp[i]*(parameters.nom_amp0_body +parameters.drive*parameters.nom_amp_drive_body)
        else :
            self.nominal_amplitudes[0:self.n_oscillators_body] = parameters.nom_amp_sat

        if((parameters.drive >= parameters.d_low_limb) and (parameters.drive <= parameters.d_high_limb)):
            self.nominal_amplitudes[self.n_oscillators_body:self.n_oscillators] = parameters.nom_amp0_limb +parameters.drive*parameters.nom_amp_drive_limb
        else:
            self.nominal_amplitudes[self.n_oscillators_body:self.n_oscillators] = parameters.nom_amp_sat
    '''
    def set_gain(self, parameters):
        """Set gain for sensory feedback"""
        self.gain= np.full(self.n_oscillators, parameters.gain_sensory)
        # Disruptions:
        if parameters.muted_sensors > 0 : 
            random.seed(parameters.seed_disruption + 10) #replicating the results when running the simulation again using the same seed.
            idx = random.sample(range(0,self.n_body_joints),parameters.muted_sensors)
            #idx = np.random.randint(0,self.n_body_joints,parameters.muted_sensors)
            for i in idx :
                self.gain[i] = 0
                self.gain[i+self.n_body_joints] = 0

    #added for 9b   
    def step(self, iteration, salamandra_data):
        """Step function called at each iteration

        Parameters
        ----------

        salamanra_data: salamandra_simulation/data.py::SalamandraData
            Contains the robot data, including network and sensors.

        gps (within the method): Numpy array of shape [9x3]
            Numpy array of size 9x3 representing the GPS positions of each link
            of the robot along the body. The first index [0-8] coressponds to
            the link number from head to tail, and the second index [0,1,2]
            coressponds to the XYZ axis in world coordinate.

        """
        gps = np.array(
            salamandra_data.sensors.links.urdf_positions()[iteration, :9],
        )
        if gps[2][0]>1.2 : 
            self.sim_parameters.drive = 4
            self.amplitude_gradient = [1.2,1.5]
            self.freq0_body = 1
            self.update(self.sim_parameters)
        if gps[2][0]<1.2 : 
            self.sim_parameters.drive = 2.7
            self.update(self.sim_parameters)


            