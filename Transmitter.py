import os
import shutil
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from coordinate_generator import *
from discrete_hydromate_single import run_discrete_hydromate_single
from filter_trajectory import *
from findpoint import *
from maps.build_folium_map import build_folium_map, build_map
from maps.build_local_3D import build_local_cartesian_map_3d
from maps.build_local_map import build_local_cartesian_map
from maps.map_common import Track
from ping_all import *
from utils_runner import *


DIM = 3

class Transmitter:
    def __init__(self, ID, gt_x, gt_y, gt_z, dt=1.0):
        self.dt = dt
        self.ID = ID
        self.steps_counter = 0
        self.gt_v = np.array([0.0, 0.0, 0.0], dtype=float)
        self.gt_pos = np.asarray([gt_x, gt_y, gt_z])

        # === Motion model parameters ===
        self.Rho = 0.0
        self.sigma = np.zeros(DIM, dtype=float)

    def set_rho(self, rho):
        """Set the autocorrelation parameter of the motion model"""
        self.Rho = rho
        
    def set_sigma(self, *args):
        """
        Set the noise standard deviations for the motion model.
        For 2D: set_sigma(sigma_x, sigma_y)
        For 3D: set_sigma(sigma_x, sigma_y, sigma_z)
        """
        if len(args) == DIM:
            self.sigma = np.array(args, dtype=float)
        else:
            raise ValueError(f"Expected {DIM} sigma values, got {len(args)}")
    
    def set_initial_velocity(self, *args):
        """
        Set the initial velocity for both true and estimated states.
        For 2D: set_initial_velocity(v_x, v_y)
        For 3D: set_initial_velocity(v_x, v_y, v_z)
        """
        if len(args) == DIM:
            vel = np.array(args, dtype=float)
            self.gt_v = vel.copy()
            self.est_v = vel.copy()
            self.v_prev = vel.copy()
        else:
            raise ValueError(f"Expected {DIM} velocity values, got {len(args)}")

    def move(self):
        self.steps_counter += 1

        # Motion model: AR(1) process for velocity
        e = np.random.normal(0, self.sigma)
        self.gt_v = self.gt_v * self.Rho + e * np.sqrt(1 - self.Rho**2)
        
        # Update true position
        self.gt_pos = self.gt_pos + self.gt_v * self.dt