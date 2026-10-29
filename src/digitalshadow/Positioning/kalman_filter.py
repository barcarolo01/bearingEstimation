import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_continuous_white_noise
from digitalshadow.positioning.point_estimation import triangulate

def measurement_covariance(floater_xy, target_xy, sigma_deg):
    sigma = np.deg2rad(sigma_deg)

    # J is the information matrix: the sum of the constraints given by each line
    J = np.zeros((2, 2))

    for h in floater_xy:
        d = target_xy - h       # Vector from the floater to the target
        r = np.linalg.norm(d)   # Floater-Target distance (estimated)
        u = d / r               # Unit vector along the bearing line
        Jn = (np.eye(2) - np.outer(u, u)) / (r * sigma) ** 2
        J += Jn

    # The covariance matrix is the inverse of the information matrix
    return np.linalg.inv(J)

def initial_state(z, R, init_speed_std):
    """
    Initialize the data structure used by Kalman Filter
    """
    # State vector [x, vx, y, vy]
    x = np.array([z[0], 0.0, z[1], 0.0]) # Init state = first measurement, init velocity = 0

    # Diagonal matrix with velocity variances on the diagonal
    P = np.diag([0.0, init_speed_std ** 2, 0.0, init_speed_std ** 2])

    # Position block
    P[0, 0] = R[0, 0]
    P[0, 2] = R[0, 1]
    P[2, 0] = R[1, 0]
    P[2, 2] = R[1, 1]
    return x, P

def estimate_TX_position(floaters, bearings, weights, dt, elevations=None,
                         sigma_deg=1.0, a_max=10, init_speed_std=10,
                         max_init_std=50.0, max_consecutive_rejects=10):
    """
    Input  
        floaters: floater positions, shape or (I,N,3)
        bearings: bearing arrays, shape (I,N)
        elevations: bearing arrays, shape (I,N)
        dt: time between two consecutive trajectory points, in seconds.
        sigma_deg: standard deviation of a single bearing, in degrees.
        a_max: largest acceleration expected for the vessel, in m/s^2.
        init_speed_std: initial velocity uncertainty, in m/s.
        max_init_std: max std deviation allowed to initialize KF, in.
        max_consecutive_rejects : maximum consecutive rejections before a KF reinitialization.

    Output
        dicts, containing a np array of shape (STEPS, 3) [lat, lon, depth] for each key:
            'raw'      : raw triangulation result
            'filtered' : Kalman filter (causal)
            'smoothed' : Kalman + RTS smoother (offline)
            'velocity' (STEPS, 2) [vx, vy] in m/s, from the smoother.
    """
    floaters = np.asarray(floaters, dtype=float)

    #weights = None

    # Compute target trajectory with minimum-RMSE triangulation
    raw = triangulate(floaters, bearings, elevations, weights)
    STEPS = len(raw)
    z_all = raw[:, :2]           # (STEPS, 2) [x, y]

    # List of z (positions) and R (position covariance)
    zs, Rs = [], []
    for i in range(STEPS):
        z, R = None, None
        if np.all(np.isfinite(z_all[i])):   # 'triangulate' method succeeded
            floater_xy = floaters[i, :, :2]
            try:
                R = measurement_covariance(floater_xy, z_all[i], sigma_deg)
                z = z_all[i]
            except np.linalg.LinAlgError: # Bearing directions are parallel
                pass
        zs.append(z)
        Rs.append(R)

    # Choose a sufficiently precise fix to start the tracking, to avoid a wrong filter initialization
    start = None
    for i in range(STEPS):
        if zs[i] is not None and np.sqrt(np.trace(Rs[i])) <= max_init_std:
            start = i
            break

    if start is None:
        print("[Kalman] No position fix precise enough to start the filter.")
        return dict(raw=raw, filtered=None, smoothed=None, velocity=None)

    # State vector has 4 elements [x, vx, y, vy], measurement vector has 2 [x,y]
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # F: motion model matrix
    kf.F = np.array([[1, dt, 0, 0],
                     [0, 1,  0, 0],
                     [0, 0,  1, dt],
                     [0, 0,  0, 1]])

    # H: measurement model. The measurement is the position itself: H extracts [x,y] from state [x, vx, y, vy].
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0]])

    # Q: process noise covariance, the uncertainty added at every prediction
    kf.Q = Q_continuous_white_noise(dim=2, dt=dt, spectral_density=a_max ** 2 * dt,block_size=2)

    # Initial state from the first reliable fix
    kf.x, kf.P = initial_state(zs[start], Rs[start], init_speed_std)

    # Filtered states and covariances, one per step from 'start' onwards
    x_f, P_f = [kf.x.copy()], [kf.P.copy()]
    
    rejects = 0 # Consecutive rejected fixes
    for z, R in zip(zs[start + 1:], Rs[start + 1:]):
        # Prediction: move the state forward by dt and increase its uncertainty
        kf.predict()

        if z is not None:
            xi = z - kf.H @ kf.x                  # Innovation
            S = kf.H @ kf.P @ kf.H.T + R          # Innovation covariance
            d2 = xi @ np.linalg.solve(S, xi)      # Squared of Mahalanobis distance (y^T S^-1 y)

            # Gate (Mahalanobis distance)
            if d2 <= 9.21: 
                # Plausible fix: correct the prediction with the measurement
                kf.update(z, R=R)
                rejects = 0
            else:
                # Outlier: ignore it and keep the prediction
                rejects += 1

                # Too many consecutive outliers: they are now considered as reliable
                if rejects >= max_consecutive_rejects and np.sqrt(np.trace(R)) <= max_init_std:
                    # Filter re-initialization
                    kf.x, kf.P = initial_state(z, R, init_speed_std)
                    rejects = 0

        # If z is None there is no measurement: the prediction is kept as a result
        x_f.append(kf.x.copy())
        P_f.append(kf.P.copy())

    x_f, P_f = np.array(x_f), np.array(P_f)

    # RTS smoother
    x_s, _, _, _ = kf.rts_smoother(x_f, P_f)

    print(start)
    filtered = np.full((STEPS, 3), np.nan)
    filtered[start:, :2] = x_f[:, [0, 2]]   # Filitered [x,y]
    filtered[:, 2] = raw[:, 2] # Depth filling

    smoothed = np.full((STEPS, 3), np.nan)
    smoothed[start:, :2] = x_s[:, [0, 2]]   # Smoothed [x,y]
    smoothed[:, 2] = raw[:, 2] # Depth filling

    velocity = np.full((STEPS, 2), np.nan)
    velocity[start:] = x_s[:, [1, 3]]  # Velocity components [vx, vy]

    return dict(raw=raw, filtered=filtered, smoothed=smoothed, velocity=velocity)