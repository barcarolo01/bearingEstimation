import numpy as np
import matplotlib.pyplot as plt

from maps.build_local_map import build_local_cartesian_map

'''
This file contians a class that simualtes a floater equipped with a IMU
'''

def local_to_geo(Center_coordinates,local_point):
    Lat_center, Lon_center = Center_coordinates[0], Center_coordinates[1]
    gt_x = local_point[..., 0]
    gt_y = local_point[..., 1]

    R = 6371000.0
    Lat = Lat_center + (gt_y / R) * (180 / np.pi)
    Lon = Lon_center + (gt_x / (R * np.cos(np.radians(Lat_center)))) * (180 / np.pi)
    depth = np.full_like(Lat, 22.0)

    return np.stack((Lat, Lon, depth), axis=-1)

def geo_to_local(Center_coordinates, geo_coordinates):
    Lats = geo_coordinates[..., 0]
    Lons = geo_coordinates[..., 1]
    Lat_center, Lon_center = Center_coordinates[0], Center_coordinates[1]

    lats = np.radians(Lats)
    lons = np.radians(Lons)
    R = 6371000.0
    x = R * (lons - np.radians(Lon_center)) * np.cos(np.radians(Lat_center))
    y = R * (lats - np.radians(Lat_center))
    z = np.full_like(x, 10.0)

    return np.stack((x, y, z), axis=-1)

def computer_local_mobility(N_STEPS,v_x_init,v_y_init,Rho,sigma_x,sigma_y):
    local_coordinates = np.zeros((N_STEPS,2))
    local_coordinates[0,:] = [0,0]

    v_x = v_x_init
    v_y = v_y_init
    v_x_y = np.sqrt(v_x**2 + v_y**2)

    for i in range(1,N_STEPS):
        e_x = np.random.normal(0,sigma_x)
        e_y = np.random.normal(0,sigma_y)
    
        v_x = v_x*Rho + e_x*np.sqrt(1 - Rho**2)
        v_y = v_y*Rho + e_y*np.sqrt(1 - Rho**2)
        
        local_coordinates[i,:] = [ local_coordinates[i-1,0] + v_x * 1, 
                                   local_coordinates[i-1,1] + v_y * 1 ]


    return np.asarray(local_coordinates)

class Floater:
    """
    This class emulates a floater, its motion model and its self-positioning algorithm based on 
    a simulated Inertial Movement Unit (IMU).
    """
    def __init__(self, gt_x, gt_y, gt_z, dt=1.0, dim=3):
        self.dt = dt
        self.dim = dim

        # True position and velocity
        if dim == 3:
            self.gt_pos = np.array([gt_x, gt_y, gt_z], dtype=float)
            self.gt_v = np.array([0.0, 0.0, 0.0], dtype=float)
        else:
            self.gt_pos = np.array([gt_x, gt_y], dtype=float)
            self.gt_v = np.array([0.0, 0.0], dtype=float)
        
        # Estimated position, velocity and acceleration
        self.est_pos = self.gt_pos.copy()
        self.est_v = self.gt_v.copy()
        self.est_a = np.zeros(self.dim, dtype=float)
        
        # Motion model parameters
        self.Rho = 0.0
        self.sigma = np.zeros(self.dim, dtype=float)
        self.gt_v = np.zeros(self.dim, dtype=float)
        
        
        # IMU parameters
        # == Contribution 1: constant bias
        self.accel_bias = np.zeros(self.dim)

        # == Contribution 2: white noise
        #self.sigma_white_noise = np.full(self.dim, 0.037 / np.sqrt(3600 * dt), dtype=float)
        self.sigma_white_noise = np.full(self.dim, 0.37 / np.sqrt(3600 * dt), dtype=float)
        
        # == Contribution 3: bias random walk (cumulative error)
        #self.sigma_bias_driving = 13e-6 * 9.81 * np.sqrt(dt / 200.0)
        self.sigma_bias_driving = 13e-4 * 9.81 * np.sqrt(dt / 200.0)
        self.bias_random_walk = np.zeros(self.dim, dtype=float) # Initially zeros
        
        
        # Previous acceleration value (used for integration)
        self.a_gt = np.zeros(self.dim, dtype=float)
        self._prev_est_a = np.zeros(self.dim, dtype=float)
    
    def set_rho(self, rho):
        """Set the autocorrelation parameter of the motion model"""
        self.Rho = rho
    
    def set_sigma(self, *args):
        """
        Set the noise standard deviations for the motion model.
        For 2D: set_sigma(sigma_x, sigma_y)
        For 3D: set_sigma(sigma_x, sigma_y, sigma_z)
        """
        if len(args) == self.dim:
            self.sigma = np.array(args, dtype=float)
        else:
            raise ValueError(f"Expected {self.dim} sigma values, got {len(args)}")
    
    def set_initial_velocity(self, *args):
        """
        Set the initial velocity for both true and estimated states.
        For 2D: set_initial_velocity(v_x, v_y)
        For 3D: set_initial_velocity(v_x, v_y, v_z)
        """
        if len(args) == self.dim:
            vel = np.array(args, dtype=float)
            self.gt_v = vel.copy()
            self.est_v = vel.copy()
            self.v_prev = vel.copy()
        else:
            raise ValueError(f"Expected {self.dim} velocity values, got {len(args)}")
    
    def set_accel_bias(self, *args):
        """Set constant acceleration bias for IMU"""
        if len(args) == self.dim:
            self.accel_bias = np.array(args, dtype=float)
        else:
            raise ValueError(f"Expected {self.dim} bias values, got {len(args)}")
    
    def move(self):
        """Perform a single motion step: update true position and IMU-based estimate"""
        
        # Keep track of the old velocity values
        self.v_prev = self.gt_v.copy()

        # Motion model: AR(1) process for velocity
        e = np.random.normal(0, self.sigma)
        self.gt_v = self.gt_v * self.Rho + e * np.sqrt(1 - self.Rho**2)
        
        # Calculate acceleration: velocity derivative
        self.a_gt = (self.gt_v - self.v_prev) / self.dt
        
        # Update true position
        self.gt_pos += self.gt_v * self.dt

        # === SIMULATE MEASURED ACCELERATION (with IMU errors) ===
        white_noise = np.random.normal(0, self.sigma_white_noise)
        bias_drift = np.random.normal(0, self.sigma_bias_driving, size=self.dim)
        
        # Accumulate random walk bias
        self.bias_random_walk += bias_drift
        
        # Measured acceleration = true acceleration + errors
        a_measured = self.a_gt + self.accel_bias + white_noise + self.bias_random_walk
        
        # === INTEGRATE ESTIMATED VELOCITY (trapezoidal rule) ===
        self.est_v += 0.5 * (self._prev_est_a + a_measured) * self.dt
        #self.est_v += a_measured * self.dt

        # === INTEGRATE ESTIMATED POSITION ===
        self.est_pos += self.est_v * self.dt
        
        # === SAVE STATE FOR NEXT ITERATION ===
        self._prev_est_a = a_measured.copy()
        self.est_a = a_measured.copy()

    def get_gt_position(self):
        """Return the true position as a numpy array"""
        return self.gt_pos.copy()

    def get_est_position(self):
        """Return the estimated position as a numpy array"""
        return self.est_pos.copy()
    
    def get_position_error(self):
        """Return the position error (true - estimated)"""
        return self.gt_pos - self.est_pos
    
    def get_position_error_magnitude(self):
        """Return the magnitude of the position error"""
        return np.linalg.norm(self.get_position_error())

    def resurface(self):
        # The estimated position is reset to the ground truth position
        self.est_pos = self.gt_pos.copy()
        '''
        # The estimated velocity is reset to the ground truth velocity
        self.est_v = self.gt_v.copy()
    
        # Accelerazions are zeroed
        self.est_a = self.a_gt
        self._prev_est_a = self.a_gt

        # Random walk (i.e. cumulative error) is zeroed
        self.bias_random_walk = np.zeros(self.dim, dtype=float)
        
        '''


if __name__ == '__main__':

    np.random.seed(5)
    Center = [12.61529, 43.37765]
    DIM = 3
    N = 900
    
    floater = Floater(gt_x=0, gt_y=0, gt_z=0.0, dt=1.0, dim  = DIM)
    floater.set_rho(0.98)
    floater.set_sigma(0.2, 0.2,0.0)
    floater.set_initial_velocity(1.0, 0.5,0.0)
    
    gt = np.zeros((N,DIM))
    est = np.zeros((N,DIM))
    for i in range(N):
        gt[i] = floater.get_gt_position()
        est[i] = floater.get_est_position()
        print(np.linalg.norm(floater.get_position_error()))
        floater.move()

    
    fig,axes= plt.subplots(1,2,figsize=(12,6))
    axes[0].plot(gt[:,0],gt[:,1],'-o',color='green',label="Ground truth")
    axes[0].plot(est[:,0],est[:,1],'-o',color='red',label="IMU estimated")
    axes[0].legend(loc="upper right", frameon=True, facecolor='white', edgecolor='grey', fontsize=9)
    axes[1].plot(np.linalg.norm(gt-est,axis=1))
    axes[0].set_xlabel("Meters")
    axes[0].set_ylabel("Meters")
    axes[1].set_xlabel("Simulation steps / seconds")
    axes[1].set_ylabel("Auto-estimated error w.r.t. ground truth [meters]")
    axes[1].set_title("Positioning error: IMU vs ground truth")
    axes[0].set_title("Estimated position vs ground truth (start at [0;0])")
    plt.show()