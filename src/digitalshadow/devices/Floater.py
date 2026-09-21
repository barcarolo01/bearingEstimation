import numpy as np
from digitalshadow.Positioning.Positioning import *
from digitalshadow.devices.Transmitter import *

# ===== DATASHEET ACCELEROMETER =====
ACCEL_BIAS_SIGMA = np.ones(3) * (4e-3)   # 4 mg da datasheet
ACCEL_SIGMA_SINGLE_WHITENOISE = np.ones(3) * (0.037 / np.sqrt(3600 * 1))
ACCEL_SIGMA_CUM_WHITENOISE =  np.ones(3) * (13e-6 * 9.81 * np.sqrt(1 / 200.0))

# ===== DATASHEET GYROSCOPE =====
GYRO_BIAS_SIGMA  = np.deg2rad(0.2) 
GYRO_SIGMA_SINGLE_WHITENOISE   = np.deg2rad(0.34) / np.sqrt(3600 * 1)
GYRO_SIGMA_CUM_WHITENOISE      = np.deg2rad(8.0 / 3600.0) * np.sqrt(1 / 200) 

# ==== E COMPASS ====
COMPASS_SIGMA_NOISE   = np.deg2rad(0.3)
COMPASS_BIAS    = np.deg2rad(2.0)
COMPASS_ALPHA   = 0.9


CONSTANT_DEPTH = True
MIN_DEPTH = 1
MAX_DEPTH = 100
CLOCK_FREQUENCY = 1000

def wrap(a):
    """Riporta un angolo nell'intervallo [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi

def Rz(psi):
    """ Builds a rotation matrix around the z axis of the floater """
    return np.array([[np.cos(psi), -np.sin(psi), 0.0],
                     [np.sin(psi),  np.cos(psi), 0.0],
                     [0.0, 0.0, 1.0]])

class Floater(Transmitter):
    """
    This class emulates a floater, its motion model and its self-positioning algorithm based on 
    a simulated Inertial Movement Unit (IMU).
    """
    def __init__(self, ID, gt_x, gt_y, gt_z, NF, dt=1.0,imu_seed=1):
        super().__init__(ID=ID, gt_x=gt_x, gt_y=gt_y, gt_z=gt_z, dt=dt)

        self.gt_v = np.zeros(3)     # Initial ground truth velocity
        self.gt_psi   = 0           # Initial ground truth heading 
        self.gt_psi_unwrapped  = 0
        self.gt_omega = 0           # Initial ground truth angular velocity

        # === RANDOM NUMBER GENERATORS ===
        self.rnd_gt = np.random.default_rng([imu_seed, ID, 0])   
        self.rnd_gyro = np.random.default_rng([imu_seed, ID, 1])    # Gyroscope
        self.rnd_accel = np.random.default_rng([imu_seed, ID, 2])   # Accelerometer
        self.rnd_compass = np.random.default_rng([imu_seed, ID, 3]) # E-compass


        self.NF = NF # Number of floaters
        self.ONGOING_RESURFACE = False
        self.ONGOING_IMMERSION = False
        self.depth_before_resurface = 0

        # History dict: maintains all values for each simulation step, used as key
        self.pos_history     = {}   # IMU only position
        self.err_history     = {}   # IMU only error
        self.pos_history_mds = {}   # IMU + MDS
        self.err_history_mds = {}   # IMU + MDS error
        self.gt_history      = {}
        self.dist_matrices = {}

        self.pos_history_compensated = {}

        # Counters
        self.steps_counter = 0      # Global counter (from the beginning of simulation)
        self.steps_since_reset = 0  # Resurface
        self.steps_since_fix   = 0  # Fix = MDS or resurface
        self._err_at_fix = 0.0      

        # == Resurface options == 
        self.resurface_velocity = 0.5       # Constant velocity in depth direction for resurface operations 
        self.incremental_resurface = True   # If False, the GPS position is acquired without changing floater depth

        # === Estimated position, velocity and acceleration ===
        self.est_a = np.zeros(3)                # Estimated acceleration

        self.est_pos = self.gt_pos.copy()       # Current estimated position (IMU only)
        self.est_pos_mds = self.gt_pos.copy()   # Current estimated position (IMU + MDS)
        self.est_error = 0                      # Current estimated error (IMU only)
        self.est_error_mds = 0                  # Current estimated error (IMU + MDS)
        
        self.est_psi = self.gt_psi              # Estimated heading
    
        self.psi_err = 0
        self.psi_err_unwrapped = 0
        
        
    
        self.pos_at_reset = self.gt_pos.copy()
        self.pos_at_fix = self.gt_pos.copy()
     
        
        # === Motion model parameters ===
        self.Rho = 1      # Autoregressive parameter for translational movements
        self.Rho_yaw = 1    # Autoregressive parameter for rotational movements (only z-axis)        
        self.sigma = np.zeros(3)    # Variance of translationval movement
        self.sigma_yaw_rate = 0     # Variance of rotational velocity
        
        # === ACCELEROMETER parameters ===
        # == Contribution 1: constant bias
        self.sigma_accel_bias = ACCEL_BIAS_SIGMA
        self.accel_bias = self.rnd_accel.normal(0,self.sigma_accel_bias)

        # == Contribution 2: white noise
        self.sigma_accel_white_noise = ACCEL_SIGMA_SINGLE_WHITENOISE
        
        # == Contribution 3: bias random walk (cumulative error)
        self.sigma_accel_bias_driving = ACCEL_SIGMA_CUM_WHITENOISE
        self.accel_bias_random_walk = np.zeros(3)


        # === GYROSCOPE parameters (z-axis only) ===
        # == Contribution 1: constant bias
        self.sigma_gyro_bias   = GYRO_BIAS_SIGMA
        self.gyro_bias  = self.rnd_gyro.normal(0, self.sigma_gyro_bias)

        # == Contribution 2: white noise
        self.sigma_gyro_white_noise  = GYRO_SIGMA_SINGLE_WHITENOISE

        # == Contribution 3: bias random walk (cumulative error)
        self.sigma_gyro_bias_driving = GYRO_SIGMA_CUM_WHITENOISE
        self.gyro_bias_random_walk   = 0

        # === E-Compass ===
        self.use_compass = False
        self.compass_bias_sigma = COMPASS_BIAS
        self.compass_bias  = self.rnd_compass.normal(0, self.compass_bias_sigma)
        self.sigma_compass = COMPASS_SIGMA_NOISE
        self.alpha_compass = COMPASS_ALPHA

        
        # == Frequency of resurface and MDS
        self.MDS_freq = np.inf
        self.RESURFACE_freq = np.inf

              
        # === Previous acceleration value (used for integration) ===
        self.a_gt = np.zeros(3)         # Ground truth acceleration
        
        self.MDS_index = []
        self.Resurface_index = []
        self.estimated_movements = []

        # == History initialization
        self.pos_history[0] = self.gt_pos.copy()                
        self.err_history[0] = 0.0
        self.pos_history_mds[0] = self.gt_pos.copy()
        self.err_history_mds[0] = 0.0
        self.gt_history[0] = self.gt_pos.copy()
        self.pos_history_compensated[0] = self.gt_pos.copy()
        self.nbr_history_pos = {}
        self.nbr_history_err = {}
        self.psi_history     = {0: self.gt_psi}
        self.psi_est_history = {0: self.est_psi}
        self.psi_err_history = {0: self.psi_err}  

        # == For ranging
        self.clk = int(np.random.rand() * 10**6) # Clock initialization: random value between 0 and 10^6
        self.obs      = {}    # {(transmitter, observer): local timestamp}
        self.round_id = 0     # Ongoing round ID
        self.events   = []
        self.mds_done = False
        self.peer_pos = {}    # Dictionary: {Floater ID: estimated position (3D)}
        self.peer_err = {}    # Dictionary: {Floater ID: estimated error}

    def _matrix_complete(self):
        """True se obs contiene tutte le celle necessarie."""
        return all((a, b) in self.obs
                   for a in range(self.NF) for b in range(self.NF))

    def _build_distance_matrix(self):
        """Estrae la matrice NxN dei ritardi (ms) dai timestamp grezzi."""
        D = np.zeros((self.NF, self.NF))
        for a in range(self.NF):
                for b in range(a + 1, self.NF):
                        d = 0.5 * ((self.obs[(a, b)] - self.obs[(a, a)]) +
                                   (self.obs[(b, a)] - self.obs[(b, b)]))
                        D[a, b] = D[b, a] = d
        return D


    def start_round(self, round_id):
        self.obs      = {}
        self.round_id = round_id
        self.mds_done = False
    
    def get_accumulated_error(self, N=None, anchor=None):
        if N is None:
            N = self.steps_since_reset
        if anchor is None:
            anchor = self.pos_at_reset
        dt = self.dt

        # Somme di potenze (forma chiusa)
        sum1 = N * (N + 1) / 2.0
        sum2 = N * (N + 1) * (2 * N + 1) / 6.0
        sum3 = (N * (N + 1) / 2.0) ** 2
        sum4 = N * (N + 1) * (2 * N + 1) * (3 * N**2 + 3 * N - 1) / 30.0

        # ---- ACCELEROMETRO: doppia integrazione -> errore di posizione ----
        sigma_pos_bias  = self.sigma_accel_bias    * dt**2 * sum1
        sigma_pos_white = self.sigma_accel_white_noise   * dt**2 * np.sqrt(sum2)
        rw_sum = (sum4 + 2 * sum3 + sum2) / 4.0
        sigma_pos_rw    = self.sigma_accel_bias_driving  * dt**2 * np.sqrt(rw_sum)

        if CONSTANT_DEPTH:
            k = 2 
        else:
            k = 3

        sigma_acc = np.sqrt(np.sum(sigma_pos_bias[:k]**2
                                 + sigma_pos_white[:k]**2
                                 + sigma_pos_rw[:k]**2))


        # ---- GIROSCOPIO: singola integrazione -> errore di heading ----
        sigma_psi_bias  = self.sigma_gyro_bias        * dt * N
        sigma_psi_white = self.sigma_gyro_white_noise * dt * np.sqrt(N)
        sigma_psi_rw    = self.sigma_gyro_bias_driving * dt * np.sqrt(sum2)

        if self.use_compass:
            sigma_psi = np.hypot(self.compass_bias_sigma,
                                 self.sigma_compass * np.sqrt(1.0 - self.alpha_compass))
        else:
            sigma_psi = np.sqrt(sigma_psi_bias**2
                              + sigma_psi_white**2
                              + sigma_psi_rw**2)

        L = float(np.linalg.norm(self.est_pos[:2] - anchor[:2]))
        sigma_heading = 2.0 * L * abs(np.sin(min(sigma_psi, np.pi) / 2.0))


        self.est_error = float(np.hypot(sigma_acc, sigma_heading))
        return self.est_error

    def move(self):
        """Perform a single motion step: update true position and IMU-based estimate"""
        # Step increment
        self.steps_counter += 1
        self.steps_since_reset += 1
        self.steps_since_fix += 1
        self.clk += CLOCK_FREQUENCY

        # Check if resurface or MDS is triggered
        TRIGGER_RANGING = self.steps_counter % self.MDS_freq == 0
        TRIGGER_RESURFACE = self.steps_counter % self.RESURFACE_freq == 0 and self.steps_counter != 1

        if TRIGGER_RESURFACE:
            if self.incremental_resurface:
                self.depth_before_resurface = self.gt_pos[2]
                self.ONGOING_RESURFACE = True
                self.ONGOING_IMMERSION = False
            else:
                 self.resurface()

        # 1. ROTATIONAL UPDATE
        e_w = self.rnd_gt.normal(0, self.sigma_yaw_rate)
        self.gt_omega = self.gt_omega * self.Rho_yaw + e_w * np.sqrt(1 - self.Rho_yaw**2)
        self.gt_psi_unwrapped = self.gt_psi_unwrapped + self.gt_omega * self.dt
        self.gt_psi = wrap(self.gt_psi + self.gt_omega * self.dt)
        

        # 2. TRANSLATIONAL UPDATE
        self.v_prev = self.gt_v.copy() # Keep track of the old velocity values
        e = np.random.normal(0, self.sigma)
        self.gt_v = self.gt_v * self.Rho + e * np.sqrt(1 - self.Rho**2)

        if self.ONGOING_RESURFACE:
            self.gt_v[2] = -np.abs(self.resurface_velocity)
        if self.ONGOING_IMMERSION:
            self.gt_v[2] = np.abs(self.resurface_velocity)

        # Calculate acceleration: velocity derivative
        self.a_gt = (self.gt_v - self.v_prev) / self.dt
        
        # Update true position
        self.gt_pos = self.gt_pos + self.gt_v * self.dt

        # 3. GYROSCOPE ESTIMATION UPDATE
        self.gyro_bias_random_walk += self.rnd_gyro.normal(0, self.sigma_gyro_bias_driving)
      
        omega_MEASURED = (self.gt_omega
                          + self.gyro_bias
                          + self.gyro_bias_random_walk
                          + self.rnd_gyro.normal(0, self.sigma_gyro_white_noise))

        psi_gyro = wrap(self.est_psi + omega_MEASURED * self.dt)
        self.est_psi = psi_gyro
        

        # 4. COMPASS (if used)
        if self.use_compass:
            psi_COMPASS = wrap(self.gt_psi
                               + self.compass_bias
                               + self.rnd_compass.normal(0, self.sigma_compass))
            
            innovation = wrap(psi_COMPASS - psi_gyro)
            self.est_psi = wrap(psi_gyro + (1.0 - self.alpha_compass) * innovation)

        self.psi_err = wrap(self.est_psi - self.gt_psi) # PSI error w.r.t. ground truth
        self.psi_err_unwrapped += (omega_MEASURED - self.gt_omega) * self.dt

        # 5. ACCELEROMETER ESTIMATION UPDATE
        # Rotate from navigation frame to body frame
        a_body = Rz(self.gt_psi).T @ self.a_gt

        
        white_noise = np.random.normal(0, self.sigma_accel_white_noise)
        bias_drift = np.random.normal(0, self.sigma_accel_bias_driving, size=3)
        
        # Accumulate random walk bias
        self.accel_bias_random_walk += bias_drift
        a_body_MEASURED = a_body + self.accel_bias + white_noise + self.accel_bias_random_walk

        # Rotate back from body frame to navigation frame
        a_MEASURED = Rz(self.est_psi) @ a_body_MEASURED

        if CONSTANT_DEPTH or self.ONGOING_IMMERSION or self.ONGOING_RESURFACE:
            a_MEASURED[2] = 0
            self.est_v[2] = 0        
        
        # === INTEGRATE ESTIMATED VELOCITY===
        self.est_v += a_MEASURED * self.dt

        # === INTEGRATE ESTIMATED POSITION ===
        estimated_movement = self.est_v * self.dt

        # Increment both estimated positions by the same quantity
        self.est_pos = self.est_pos+estimated_movement
        self.est_pos_mds = self.est_pos_mds+estimated_movement


        if self.ONGOING_IMMERSION:
            if self.gt_pos[2] > self.depth_before_resurface:
                self.gt_v[2] = 0
                self.gt_pos[2] = self.depth_before_resurface
                self.ONGOING_IMMERSION = False
                print(f"F{self.ID}-T{self.steps_counter} FINITO")
    
        if self.gt_pos[2] > MAX_DEPTH:
            self.gt_pos[2] = MAX_DEPTH
        if self.gt_pos[2] < MIN_DEPTH:
            self.gt_pos[2] = MIN_DEPTH
            if self.ONGOING_RESURFACE:
                self.resurface()
                print(f"Floater {self.ID} - Timestamp {self.steps_counter}: Resurface completed")
                self.ONGOING_RESURFACE = False
                self.ONGOING_IMMERSION = True
                
        if self.est_pos[2] > MAX_DEPTH:
            self.est_pos[2] = MAX_DEPTH
        if self.est_pos[2] < MIN_DEPTH:
            self.est_pos[2] = MIN_DEPTH

        if self.est_pos_mds[2] > MAX_DEPTH:
                self.est_pos_mds[2] = MAX_DEPTH
        if self.est_pos_mds[2] < MIN_DEPTH:
            self.est_pos_mds[2] = MIN_DEPTH
        
        self.estimated_movements.append(estimated_movement)
        self.est_error = self.get_accumulated_error()
    
        self.est_error_mds = float(np.hypot(self._err_at_fix,
                                            self.get_accumulated_error(self.steps_since_fix,
                                            self.pos_at_fix)))
        

        # If the observation matrix is complete, run MDS
        if not self.mds_done and self._matrix_complete():
            D_ms = self._build_distance_matrix()
            D_m  = D_ms * 1500.0 / 1000.0
            pos_all, err_all = self._assemble_priors()

            self.nbr_history_pos[self.steps_counter] = pos_all.copy()
            self.nbr_history_err[self.steps_counter] = err_all.copy()
            self.MDS_index.append(self.steps_counter)
            self.dist_matrices[self.steps_counter] = D_m

            new_pos, new_err = estimate(pos_all, err_all, D_m)

            self.est_pos_mds   = new_pos[self.ID].copy()
            self.est_error_mds = float(new_err[self.ID])
            self.steps_since_fix = 0
            self.pos_at_fix = self.est_pos_mds.copy()
            self._err_at_fix = self.est_error_mds
            self.mds_done = True

        # History update
        self.pos_history[self.steps_counter]     = self.est_pos.copy()
        self.err_history[self.steps_counter]     = self.est_error
        self.pos_history_mds[self.steps_counter] = self.est_pos_mds.copy()
        self.err_history_mds[self.steps_counter] = self.est_error_mds
        self.psi_history[self.steps_counter]     = self.gt_psi
        self.psi_est_history[self.steps_counter] = self.est_psi
        self.psi_err_history[self.steps_counter] = self.psi_err
        self.pos_history_compensated[self.steps_counter] = self.est_pos.copy()
        self.gt_history[self.steps_counter]      = self.gt_pos.copy()
            
        # Save the current acceleration (for the next iteration)
        self.est_a = a_MEASURED.copy()

        return TRIGGER_RANGING

    # === RANGING METHODS: To be executed upon transmission and receptions of packets ===
    def on_transmit(self, t_tx, tx_idx):
        self.obs.setdefault((tx_idx, tx_idx), t_tx)
        return {
            "round_id": self.round_id,
            "obs": dict(self.obs),
            "src_pos": np.asarray(self.est_pos_mds, dtype=float).copy(),
            "src_err": float(self.est_error_mds),
        }

    def on_receive(self, t_rx, tx_idx, my_idx, payload, src_ID):
        self.events.append({
                "t_local": t_rx,
                "type": "RX",
                "src": src_ID,
                "payload": payload,
        })

        if payload["round_id"] != self.round_id:
                return

        # Possible substitution: sender's estimation is always newer than receiver's
        self.peer_pos[src_ID] = payload["src_pos"]
        self.peer_err[src_ID] = payload["src_err"]

        self.obs.setdefault((tx_idx, my_idx), t_rx)
        for k, v in payload["obs"].items():
                self.obs.setdefault(k, v)

    def _assemble_priors(self):
        """Costruisce pos_all/err_all da peer_pos, con fallback per i mancanti."""
        pos_all = np.zeros((self.NF, len(self.est_pos_mds)))
        err_all = np.zeros(self.NF)

        for j in range(self.NF):
                if j == self.ID:
                        pos_all[j] = self.est_pos_mds
                        err_all[j] = self.est_error_mds
                elif j in self.peer_pos:
                        pos_all[j] = self.peer_pos[j]
                        err_all[j] = self.peer_err[j]
                else:
                        pos_all[j] = self.est_pos_mds       # nessuna info
                        err_all[j] = 1e6                    # errore enorme
        return pos_all, err_all
    
    def return_results(self, fuse=True):
        """
        Computes estimated positions and errors performing the backward analysis
        and returns the results as a set of dicts: name -> (pos (I+1,dim), err (I+1,))
        """

        '''
        rev_imu = reverse_node(self, self.pos_history, self.err_history,
                            use_mds=False, gps_sigma=0, fuse=fuse,
                            nbr_history_pos=self.nbr_history_pos, nbr_history_err=self.nbr_history_err)
        
        rev_mds = reverse_node(self, self.pos_history_mds, self.err_history_mds,
                            use_mds=True,  gps_sigma=0, fuse=fuse,
                                nbr_history_pos=self.nbr_history_pos, nbr_history_err=self.nbr_history_err)
        '''

        self.pos_history_compensated = {k: np.asarray(v, float).copy()
                                    for k, v in self.pos_history.items()}
        for idx, p in enumerate(self.Resurface_index):
            res_prec = self.Resurface_index[idx - 1] if idx > 0 else 0

            if p - res_prec < 2:          # intervallo troppo corto
                    continue

            # Drift osservato: l'ultimo step prima che resurface() sovrascriva
            DELTA = self.pos_history[p-1] - self.gt_history[p-1]

            span = (p - 1) - res_prec
            for k in range(res_prec, p):
                    w = (k - res_prec) / span          # 0 a res_prec, 1 a p-1
                    self.pos_history_compensated[k] = self.pos_history[k] - DELTA * w**2
        
        return {
            'imu':             _dict_to_array(self.pos_history),
            #'imu_rev':         _dict_to_array(rev_imu[0]),
            'imu_mds':         _dict_to_array(self.pos_history_mds),
            #'imu_mds_rev':     _dict_to_array(rev_mds[0]),
            'imu_compensated': _dict_to_array(self.pos_history_compensated),
            'gt':              _dict_to_array(self.gt_history),

            'psi_gt':   _dict_to_array_scalar(self.psi_history),      
            'psi_est':  _dict_to_array_scalar(self.psi_est_history),  # Euto-estimated
            'psi_err':  _dict_to_array_scalar(self.psi_err_history),  # Estimated - groundtruth, in [-180, 180)
        }

    def resurface(self):
        # The estimated position is reset to the ground truth position
        self.est_pos = self.gt_pos.copy()
        self.est_pos_mds = self.gt_pos.copy()
        self.pos_at_fix = self.gt_pos.copy()
        self.pos_at_reset = self.gt_pos.copy()

        # The estimated velocity is reset to the ground truth velocity
        self.est_v = self.gt_v.copy()

        # Accelerazions are zeroed
        self.est_a = self.a_gt.copy()

        # Random walk (i.e. cumulative error) is zeroed
        self.accel_bias_random_walk = np.zeros(3, dtype=float)

        # Reset all counters and errors
        self.steps_since_reset = 0
        self.steps_since_fix   = 0
        self._err_at_fix = 0.0
        self.est_error = 0.0
        self.est_error_mds = 0.0

        self.est_psi = self.gt_psi
        self.psi_err = 0.0
        self.gyro_bias_random_walk = 0.0

        self.pos_history[self.steps_counter]     = self.est_pos.copy()
        self.err_history[self.steps_counter]     = 0
        self.pos_history_mds[self.steps_counter] = self.est_pos_mds.copy()
        self.err_history_mds[self.steps_counter] = 0
        self.psi_history[self.steps_counter] = self.gt_psi
        self.psi_est_history[self.steps_counter] = self.est_psi
        self.psi_err_history[self.steps_counter] = 0
        self.Resurface_index.append(self.steps_counter)
                 
    def set_initial_velocity(self, *args):
        super().set_initial_velocity(*args)
        self.est_v = self.gt_v.copy()

    def set_accel_bias(self, *args):
        """Set constant acceleration bias for IMU"""
        if len(args) == 3:
            self.accel_bias = np.array(args, dtype=float)
        else:
            raise ValueError(f"Expected 3 bias values, got {len(args)}")

        
def _dict_to_array(d):
    keys = sorted(d)
    return np.asarray([d[k] for k in keys], dtype=float)

def _dict_to_array_scalar(d, deg=True):
    """dict {step: float} -> np.array (I+1,) ordinato per step."""
    keys = sorted(d.keys())
    arr = np.array([d[k] for k in keys], dtype=float)
    return np.rad2deg(arr) if deg else arr