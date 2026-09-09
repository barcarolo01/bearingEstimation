import numpy as np
from Positioning import estimate, move_error, reverse_node

# ===== DATASHEET PARAMETERS =====
IMU_ACCEL_BIAS = np.ones(3) * 0
IMU_SIGMA_WHITENOISE = np.ones(3) * (0.037 / np.sqrt(3600 * 1))
IMU_SIGMA_BIAS_DRIVING =  np.ones(3) * (13e-6 * 9.81 * np.sqrt(1 / 200.0))

# ===== FREQUENCY OF MDS (simulation steps) =====
MDS_FREQ = 999999

def distance_matrix(obs, N):
    D = np.full((N, N), np.nan)
    for a in range(N):
        for b in range(a+1, N):
            try:
                v = 0.5*((obs[(a,b)] - obs[(a,a)]) + (obs[(b,a)] - obs[(b,b)]))
            except KeyError:
                continue
            D[a,b] = D[b,a] = v
    return D


class Floater:
    """
    This class emulates a floater, its motion model and its self-positioning algorithm based on 
    a simulated Inertial Movement Unit (IMU).
    """
    def __init__(self, ID, gt_x, gt_y, gt_z, NF, dt=1.0, dim=3):
        self.dt = dt
        self.dim = dim
        self.ID = ID
        self.NF = NF

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

        # === True position and velocity ===
        if dim == 3:
            self.gt_pos = np.array([gt_x, gt_y, gt_z], dtype=float)
            self.gt_v = np.array([0.0, 0.0, 0.0], dtype=float)
        else:
            self.gt_pos = np.array([gt_x, gt_y], dtype=float)
            self.gt_v = np.array([0.0, 0.0], dtype=float)
        
        # === Estimated position, velocity and acceleration ===
        self.est_pos = self.gt_pos.copy()       # Current estimated position (IMU only)
        self.est_pos_mds = self.gt_pos.copy()   # Current estimated position (IMU + MDS)
        self.est_error = 0                      # Current estimated error (IMU only)
        self.est_error_mds = 0                  # Current estimated error (IMU + MDS)
        self.est_v = self.gt_v.copy()
        self.est_a = np.zeros(self.dim, dtype=float)
        
        # === Motion model parameters ===
        self.Rho = 0.0
        self.sigma = np.zeros(self.dim, dtype=float)
        self.gt_v = np.zeros(self.dim, dtype=float)
        
        # === IMU parameters ===
        # == Contribution 1: constant bias
        self.accel_bias = IMU_ACCEL_BIAS

        # == Contribution 2: white noise
        self.sigma_white_noise = IMU_SIGMA_WHITENOISE
        
        # == Contribution 3: bias random walk (cumulative error)
        self.sigma_bias_driving = IMU_SIGMA_BIAS_DRIVING
        self.bias_random_walk = np.zeros(self.dim, dtype=float) # Initially zeros
                
        # === Previous acceleration value (used for integration) ===
        self.a_gt = np.zeros(self.dim, dtype=float)
        self._prev_est_a = np.zeros(self.dim, dtype=float)
        
        self.MDS_index = []
        self.Resurface_index = []
        self.estimated_movements = []

        # History initialization
        self.pos_history[0] = self.gt_pos.copy()
        self.err_history[0] = 0.0
        self.pos_history_mds[0] = self.gt_pos.copy()
        self.err_history_mds[0] = 0.0
        self.gt_history[0] = self.gt_pos.copy()
        self.pos_history_compensated[0] = self.gt_pos.copy()
        self.nbr_hist_pos = {}
        self.nbr_hist_err = {}

        # FOR RANGING
        self.clk = int(np.random.rand() * 10**6)
        #self.clk = 0

        self.obs      = {}    # {(trasmettitore, osservatore): timestamp locale}
        self.round_id = 0     # round attualmente seguito
        self.events   = []


        self.events = []
        self.mds_done = False
        self.dist_matrix  = None
        self.dist_history = []
        self.peer_pos = {}    # {ID: np.array posizione stimata}
        self.peer_err = {}    # {ID: float errore stimato}

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
    
    def get_accumulated_error(self):
        N = self.steps_since_reset
        # Sum of powers (closed formm)
        sum2 = N * (N + 1) * (2 * N + 1) / 6.0                      # sum of j^2
        sum3 = (N * (N + 1) / 2.0) ** 2                              # sum of j^3
        sum4 = N * (N + 1) * (2 * N + 1) * (3 * N**2 + 3 * N - 1) / 30.0  # sum of j^4

        # 1) Costant bias  -> tends to N^2
        sigma_pos_bias = np.abs(IMU_ACCEL_BIAS) * self.dt**2 * N * (N + 1) / 2.0

        # 2) Single white noise contribution -> tends to N^1.5
        sigma_pos_white = IMU_SIGMA_WHITENOISE * self.dt**2 * np.sqrt(sum2)

        # 3) Cumulative white noise contribution -> tends to N^2.5
        rw_sum = (sum4 + 2 * sum3 + sum2) / 4.0
        sigma_pos_rw = IMU_SIGMA_BIAS_DRIVING * self.dt**2 * np.sqrt(rw_sum)

        # RSS combination
        sigma_pos_total = np.sqrt(sigma_pos_bias**2 + sigma_pos_white**2 + sigma_pos_rw**2)

        self.est_error = float(np.linalg.norm(sigma_pos_total))

    def increment_clock(self):
        self.clk += 1000

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

        # la stima del mittente è sempre più fresca di quella che avevo
        self.peer_pos[src_ID] = payload["src_pos"]
        self.peer_err[src_ID] = payload["src_err"]

        self.obs.setdefault((tx_idx, my_idx), t_rx)
        for k, v in payload["obs"].items():
                self.obs.setdefault(k, v)

    def move(self, gt_all=None, others_pos=None, others_err=None):
        """Perform a single motion step: update true position and IMU-based estimate"""
        # Step increment
        self.steps_counter += 1
        self.steps_since_reset += 1
        self.steps_since_fix += 1
        self.clk += 1000
        TRIGGER_RANGING = self.steps_counter % MDS_FREQ == 0
        
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
        estimated_movement = self.est_v * self.dt

        # Increment both estimated positions by the same quantity
        self.est_pos += estimated_movement
        self.est_pos_mds += estimated_movement
        self.estimated_movements.append(estimated_movement)
        self.est_error = move_error(self.steps_since_reset, self.dt)
        self.est_error_mds = float(np.hypot(self._err_at_fix, move_error(self.steps_since_fix, self.dt)))

        if not self.mds_done and self._matrix_complete():
            D_ms = self._build_distance_matrix()
            D_m  = D_ms * 1500.0 / 1000.0
            pos_all, err_all = self._assemble_priors()

            self.nbr_hist_pos[self.steps_counter] = pos_all.copy()
            self.nbr_hist_err[self.steps_counter] = err_all.copy()
            self.MDS_index.append(self.steps_counter)
            self.dist_matrices[self.steps_counter] = D_m

            new_pos, new_err = estimate(pos_all, err_all, D_m)
            self.est_pos_mds   = new_pos[self.ID].copy()
            self.est_error_mds = float(new_err[self.ID])
            self.steps_since_fix = 0
            self._err_at_fix = self.est_error_mds
            self.mds_done = True

        # History update
        self.pos_history[self.steps_counter]     = self.est_pos.copy()
        self.err_history[self.steps_counter]     = self.est_error
        self.pos_history_mds[self.steps_counter] = self.est_pos_mds.copy()
        self.err_history_mds[self.steps_counter] = self.est_error_mds

        self.pos_history_compensated[self.steps_counter] = self.est_pos.copy()

        self.gt_history[self.steps_counter]      = self.gt_pos.copy()
            
        # Save the current acceleration (for the next iteration)
        self._prev_est_a = a_MEASURED.copy()
        self.est_a = a_MEASURED.copy()



        return TRIGGER_RANGING


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
    
    def return_results(self, gps_sigma=0.0, fuse=True):
        """
        Computes estimated positions and errors performing the backward analysis
        and returns the results as a set of dicts: name -> (pos (I+1,dim), err (I+1,))
        """

        rev_imu = reverse_node(self, self.pos_history, self.err_history,
                            use_mds=False, gps_sigma=gps_sigma, fuse=fuse,
                            nbr_hist_pos=self.nbr_hist_pos, nbr_hist_err=self.nbr_hist_err)
        
        rev_mds = reverse_node(self, self.pos_history_mds, self.err_history_mds,
                            use_mds=True,  gps_sigma=gps_sigma, fuse=fuse,
                                nbr_hist_pos=self.nbr_hist_pos, nbr_hist_err=self.nbr_hist_err)

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




        self.pos_history_compensated_bis = {k: np.asarray(v, float).copy()
                                            for k, v in self.pos_history_mds.items()}
        for idx, p in enumerate(self.MDS_index):
            res_prec = self.MDS_index[idx - 1] if idx > 0 else 0

            if p - res_prec < 2:          # intervallo troppo corto
                    continue

            # Drift osservato: l'ultimo step prima che resurface() sovrascriva
            DELTA = self.pos_history[p-1] - self.gt_history[p-1]

            span = (p - 1) - res_prec
            for k in range(res_prec, p):
                    w = (k - res_prec) / span          # 0 a res_prec, 1 a p-1
                    self.pos_history_compensated_bis[k] = self.pos_history_mds[k] - DELTA * w**2
            
                            
        return {
            'imu':             self._to_arrays(self.pos_history,     self.err_history),
            'imu_rev':         self._to_arrays(*rev_imu),
            'imu_mds':         self._to_arrays(self.pos_history_mds, self.err_history_mds),
            'imu_mds_rev':     self._to_arrays(*rev_mds),
            'imu_compensated':     self._to_arrays(self.pos_history_compensated, self.err_history_mds),
            'imu_compensated_bis':     self._to_arrays(self.pos_history_compensated_bis, self.err_history_mds),
            'gt':              self._to_arrays(self.gt_history,{k: 0.0 for k in self.gt_history})[0],
        }


    def _to_arrays(self, pos_d, err_d):
        I = self.steps_counter
        pos = np.zeros((I + 1, self.dim))
        err = np.zeros(I + 1)
        for k in range(I + 1):
            pos[k] = np.asarray(pos_d[k], dtype=float).reshape(self.dim)
            err[k] = float(np.linalg.norm(err_d[k]))
        return pos, err


    def resurface(self):
        # The estimated position is reset to the ground truth position
        self.est_pos = self.gt_pos.copy()
        self.est_pos_mds = self.gt_pos.copy()

        # The estimated velocity is reset to the ground truth velocity
        self.est_v = self.gt_v.copy()

        # Accelerazions are zeroed
        self.est_a = self.a_gt
        self._prev_est_a = self.a_gt

        # Random walk (i.e. cumulative error) is zeroed
        self.bias_random_walk = np.zeros(self.dim, dtype=float)

        # Reset all counters and errors
        self.steps_since_reset = 0
        self.steps_since_fix   = 0
        self._err_at_fix = 0.0
        self.est_error = 0.0
        self.est_error_mds = 0.0

        self.pos_history[self.steps_counter]     = self.est_pos.copy()
        self.err_history[self.steps_counter]     = 0.0
        self.pos_history_mds[self.steps_counter] = self.est_pos_mds.copy()
        self.err_history_mds[self.steps_counter] = 0.0
        self.Resurface_index.append(self.steps_counter)
                    
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
    
    def _forward_mds(self, D, others_pos, others_err):
        pos_all = np.asarray(others_pos, dtype=float).copy()
        err_all = np.asarray(others_err, dtype=float).copy()
        pos_all[self.ID] = self.est_pos_mds
        err_all[self.ID] = self.est_error_mds
        new_pos, new_err = estimate(pos_all, err_all, D)
        self.est_pos_mds   = new_pos[self.ID].copy()
        self.est_error_mds = float(new_err[self.ID])
        self.steps_since_fix = 0
        self._err_at_fix = self.est_error_mds
