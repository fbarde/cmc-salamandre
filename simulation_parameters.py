"""Simulation parameters"""
import numpy as np

class SimulationParameters:
    """Simulation parameters"""

    def __init__(self, **kwargs):
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = 8
        self.n_legs_joints = 4
        self.duration = 30
        self.timestep = 1e-2
        self.phase_lag = None
        self.amplitude_gradient = None

        #weights
        self.body_weights = 10
        self.contra_body_weights = 10
        self.body_limb_weights = 30
        self.limb_weights = 10

        #frequency params
        self.freq0_body = 0.3
        self.freq_drive_body = 0.2
        self.f_sat_body = 0

        self.freq0_limb = 0
        self.freq_drive_limb = 0.2
        self.f_sat_limb = 0

        #drive
        self.drive_factor = 0.3
        self.drive_array = np.array([0]) #self.drive_factor*np.arange(0,self.duration,self.timestep)#3
        self.drive = 0

        self.d_low_body = 1
        self.d_high_body = 5

        self.d_low_limb = 1
        self.d_high_limb = 3

        #phase bias
        self.phase_bias_body_up = np.pi/4
        self.phase_bias_body_down = -np.pi/4
        self.phase_bias_body_contralat = np.pi
        self.phase_bias_limb_body = np.pi
        self.phase_bias_limb = np.pi

        #amplitude parameters
        self.amp_rates = 20

        self.nom_amp0_body = 0.196
        self.nom_amp_drive_body = 0.065

        self.nom_amp0_limb = 0.131
        self.nom_amp_drive_limb = 0.131

        self.nom_amp_sat = 0

        #amplitudes of the first and last oscillator of the spine.
        self.amp_first_last = [1,1]

        # Gain for the sensory feedback w_fb
        self.gain_sensory = 0

        # DISRUPTIONS:
        # parameter to mute sensors (loads=0, wbf=0) : 
        # max value: 8
        self.muted_sensors = 0

        # parameter to remove_couplings (w=0): 
        # max value: 7
        self.remove_couplings = 0

        # parameter to mute oscillators (f=0) : 
        # max value: 8
        self.muted_oscillators = 0

        # Parameter to set the seed when apply disruptions
        self.seed_disruption = 1

        #parameter for turning : <1 to turn right, >1 to turn left
        self.turn = 1
        
        #self.amphibious = False
        # Feel free to add more parameters (ex: MLR drive)
        # self.drive_mlr = ...
        # ...
        # Update object with provided keyword arguments
        # NOTE: This overrides the previous declarations
        self.__dict__.update(kwargs)

    def update_drive(self,iteration):
        self.drive_array = np.append(self.drive_array, self.drive_array[iteration-1]+self.drive_factor*self.timestep)
        self.drive = self.drive_array[iteration-1]

