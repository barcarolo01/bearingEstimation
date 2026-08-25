import numpy as np
import matplotlib.pyplot as plt

# DATASHEET PARAMETERS
IMU_ACCEL_BIAS = np.ones(3) * 0         

#IMU_SIGMA_WHITENOISE = np.ones(3) * (0.037 / np.sqrt(3600 * 1))
IMU_SIGMA_WHITENOISE = np.ones(3) * (0.5 / np.sqrt(3600 * 1))

#IMU_SIGMA_BIAS_DRIVING =  np.ones(3) * (13e-6 * 9.81 * np.sqrt(1 / 200.0))
IMU_SIGMA_BIAS_DRIVING =  np.ones(3) * (13e-4 * 9.81 * np.sqrt(1 / 200.0))


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
        self.accel_bias = IMU_ACCEL_BIAS

        # == Contribution 2: white noise
        self.sigma_white_noise = IMU_SIGMA_WHITENOISE
        
        
        # == Contribution 3: bias random walk (cumulative error)
        self.sigma_bias_driving = IMU_SIGMA_BIAS_DRIVING

        self.bias_random_walk = np.zeros(self.dim, dtype=float) # Initially zeros
        self.sigma_bias_driving = np.zeros(self.dim)
        
        
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
        
        # MEASURED acceleration = true acceleration + errors
        a_MEASURED = self.a_gt + self.accel_bias + white_noise + self.bias_random_walk
        
        # === INTEGRATE ESTIMATED VELOCITY (trapezoidal rule) ===
        #self.est_v += 0.5 * (self._prev_est_a + a_MEASURED) * self.dt
        self.est_v += a_MEASURED * self.dt

        # === INTEGRATE ESTIMATED POSITION ===
        self.est_pos += self.est_v * self.dt
        
        # === SAVE STATE FOR NEXT ITERATION ===
        self._prev_est_a = a_MEASURED.copy()
        self.est_a = a_MEASURED.copy()

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
        
        # The estimated velocity is reset to the ground truth velocity
        self.est_v = self.gt_v.copy()
    
        # Accelerazions are zeroed
        self.est_a = self.a_gt
        self._prev_est_a = self.a_gt

        # Random walk (i.e. cumulative error) is zeroed
        self.bias_random_walk = np.zeros(self.dim, dtype=float)


if __name__ == '__main__':
    np.random.seed(999)    
    DIM = 3
    N = 1000
    
    floater = Floater(gt_x=0, gt_y=0, gt_z=0.0, dt=1.0, dim = DIM)
    floater.set_rho(0.992)
    floater.set_sigma(0.2, 0.2, 0.0)
    floater.set_initial_velocity(0.5, 0.5, 0.0)
    step = 300
    gt = np.zeros((N,DIM))
    est = np.zeros((N,DIM))
    for i in range(N):
        if i % step == 0: 
            floater.resurface()

        gt[i] = floater.get_gt_position()
        est[i] = floater.get_est_position()
        print(np.linalg.norm(floater.get_position_error()))
        floater.move()

   # ========== PLOTTING ==========

    FONTSIZE = 20

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))

    # Gli indici in cui avviene il resurface()
    indices = list(range(0, len(est), step))


    # ============================================================
    # 1. Traiettoria stimata
    # ============================================================
    #
    # Tutta la traiettoria è continua, TRANNE il segmento
    # immediatamente precedente a ogni resurface:
    #
    #   199 -> 200
    #   399 -> 400
    #   599 -> 600
    #   ...
    #
    # Questo segmento rappresenta il "riposizionamento" della
    # traiettoria stimata sulla posizione reale.
    # ============================================================

    for i in range(len(est) - 1):

        # Il segmento i -> i+1 è quello che porta al resurface
        if (i + 1) % step == 0:
            linestyle = "--"
            lw=0.5
        else:
            linestyle = "-"
            lw  = 2

        axes.plot(
            np.abs(est[i:i+2, 0]),
            np.abs(est[i:i+2, 1]),
            color="blue",
            linewidth=lw,
            linestyle=linestyle,
            
        )


    # Label della traiettoria stimata
    axes.plot(
        [],
        [],
        color="blue",
        linewidth=2,
        linestyle="-",
        label="IMU estimated",
    )


    # ============================================================
    # 2. Punti in corrispondenza dei resurface
    # ============================================================
    #
    # Visualizziamo solo i punti in cui la traiettoria viene
    # riposizionata, senza aggiungere forzatamente l'ultimo punto.
    # ============================================================


    axes.plot(
        np.abs(est[indices, 0]),
        np.abs(est[indices, 1]),
        "o",
        color="blue",
        markersize=10,
    )
    axes.plot(
        np.abs(est[indices[0], 0]),
        np.abs(est[indices[0], 1]),
        "o",
        color="red",
        markersize=10,
    )



    # ============================================================
    # 3. Ground truth
    # ============================================================

    axes.plot(
        np.abs(gt[:, 0]),
        np.abs(gt[:, 1]),
        "o",
        color="red",
        label="Ground truth",
        markersize=3,
    )


    # ============================================================
    # 4. Etichetta del primo punto
    # ============================================================

    if len(indices) > 0:
        idx = indices[0]

        axes.annotate(
            f"t={idx} s",
            (np.abs(est[idx, 0]), np.abs(est[idx, 1])),
            textcoords="offset points",
            xytext=(25, -10),
            fontsize=FONTSIZE,
            color="red",
        )


    # ============================================================
    # 5. Impostazioni grafiche
    # ============================================================

    axes.set_xlabel("Meters", fontsize=FONTSIZE)
    axes.set_ylabel("Meters", fontsize=FONTSIZE)

    axes.tick_params(
        axis="both",
        which="major",
        labelsize=FONTSIZE,
    )

    axes.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="grey",
        fontsize=FONTSIZE,
    )

    axes.grid(True, linestyle="--", alpha=0.5)

    plt.savefig("ADIS16470_GT_vs_estimated_resurface.png")
    plt.show()


    # ============================================================
    # 6. Errore di posizione
    # ============================================================

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))

    axes.plot(
        np.linalg.norm(gt - est, axis=1),
        linewidth=3.0,
    )

    axes.grid(True, linestyle="--", alpha=0.5)

    axes.set_xlabel(
        "Simulation steps / seconds",
        fontsize=FONTSIZE,
    )

    axes.set_ylabel(
        "Meters",
        fontsize=FONTSIZE,
    )

    axes.tick_params(
        axis="both",
        which="major",
        labelsize=FONTSIZE,
    )

    plt.savefig("ADIS16470_accumulated_error_resurface.png")
    plt.show()