from digitalshadow.Positioning.coordinate_generator import *
from digitalshadow.Positioning.filter_trajectory import *
from digitalshadow.Positioning.point_estimation import *
from digitalshadow.underwater_sim.floater_ping import *
from utils.utils import *

'''
This file constains the class Transmitter.
It simply models an object free to mode in a ocean current with configurable parameters.
Class Floater, defined in its own file, derives from Transmitter.
'''

class Transmitter:
    def __init__(self, ID, gt_x, gt_y, gt_z, dt=1.0):
        self.dt = dt
        self.ID = ID
        self.steps_counter = 0
        self.gt_v = np.array([0.0, 0.0, 0.0])
        self.gt_pos = np.asarray([gt_x, gt_y, gt_z])

        # === Motion model parameters ===
        self.Rho = 0.0
        self.sigma = np.zeros(3)

    def set_rho(self, rho):
        """Set the autocorrelation parameter of the motion model"""
        self.Rho = rho
        
    def set_sigma(self, *args):
        """
        Set the noise standard deviations for the motion model.
        For 2D: set_sigma(sigma_x, sigma_y)
        For 3D: set_sigma(sigma_x, sigma_y, sigma_z)
        """
        if len(args) == 3:
            self.sigma = np.array(args, dtype=float)
        else:
            raise ValueError(f"Expected 3 sigma values, got {len(args)}")
    
    def set_initial_velocity(self, *args):
        """
        Set the initial velocity for both true and estimated states.
        For 2D: set_initial_velocity(v_x, v_y)
        For 3D: set_initial_velocity(v_x, v_y, v_z)
        """
        if len(args) == 3:
            vel = np.array(args, dtype=float)
            self.gt_v = vel.copy()
            self.est_v = vel.copy()
            self.v_prev = vel.copy()
        else:
            raise ValueError(f"Expected 3 velocity values, got {len(args)}")

    def move(self):
        self.steps_counter += 1

        # Motion model: AR(1) process for velocity
        e = np.random.normal(0, self.sigma)
        self.gt_v = self.gt_v * self.Rho + e * np.sqrt(1 - self.Rho**2)
        
        # Update true position
        self.gt_pos = self.gt_pos + self.gt_v * self.dt