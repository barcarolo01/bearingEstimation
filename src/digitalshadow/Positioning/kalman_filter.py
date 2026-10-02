import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_continuous_white_noise
from digitalshadow.Positioning.point_estimation import find_points

def measurement_covariance(floater_xy, target_xy, sigma_deg):
    sigma = np.deg2rad(sigma_deg)

    # J is the information matrix: the sum of the constraints given by each line
    J = np.zeros((2, 2))

    for h in floater_xy:
        d = target_xy - h       # Cector from the floater to the target
        r = np.linalg.norm(d)   # Floater-Target distance (estimated)
        u = d / r               # Unit vector along the bearing line
        Jn = (np.eye(2) - np.outer(u, u)) / (r * sigma) ** 2
        J += Jn

    # The covariance is the inverse of the information
    return np.linalg.inv(J)

def initial_state(z, R, init_speed_std):
    """
    State and covariance used to start (or restart) the filter from a fix.

    The position is taken from the fix z, with the same uncertainty R as the
    fix itself. The velocity is unknown: it is set to zero with a large
    standard deviation init_speed_std [m/s], so that the filter learns it from
    the following measurements. Position and velocity errors are uncorrelated.
    """
    # State vector [x, vx, y, vy]
    x = np.array([z[0], 0.0, z[1], 0.0])

    # Velocity variances on the diagonal (positions filled below)
    P = np.diag([0.0, init_speed_std ** 2, 0.0, init_speed_std ** 2])

    # Position block (rows/columns 0 and 2 of the state) = covariance of the fix
    P[np.ix_([0, 2], [0, 2])] = R
    return x, P

def kalman_track(floaters, bearings, dt, elevation_array=None, sigma_deg=1.0,
                 a_max=0.5, init_speed_std=15.0, max_init_std=50.0,
                 gate=9.21, max_consecutive_rejects=3):
    """
    Parameters
    ----------
    floaters, bearings, elevation_array : same as find_points.
    dt       : time between two consecutive trajectory points [s].
    sigma_deg: standard deviation of a single bearing [deg].
    a_max    : largest acceleration expected for the vessel [m/s^2].
    init_speed_std : initial velocity uncertainty [m/s] (velocity starts at 0).
    max_init_std   : the filter starts at the first fix whose position
                     standard deviation is below this value [m].
    gate     : a fix is rejected as an outlier if its squared Mahalanobis
               distance from the prediction exceeds this value
               (9.21 = 99% quantile of a chi-square with 2 dof).
    max_consecutive_rejects : after this many consecutive rejected fixes the
               target is considered lost and the filter restarts from the
               current fix (if precise enough).

    Returns
    -------
    dict of (STEPS, 3) arrays [lat, lon, depth]:
        'raw'      : find_points output
        'filtered' : Kalman filter (causal, usable in real time)
        'smoothed' : Kalman + RTS smoother (offline)
    plus 'velocity' (STEPS, 2) [vx, vy] in m/s from the smoother.
    Depth is taken from find_points. Steps before the first reliable fix are NaN.
    """
    floaters = np.asarray(floaters, dtype=float)

    # Compute target trajectory with minimum-RMSE triangulation
    raw = find_points(floaters, bearings, elevation_array)
    STEPS = len(raw)

    # The filter works in metres: local plane centred on the mean floater position
    # TODO: Remove
    center = np.nanmean(floaters[..., :2].reshape(-1, 2), axis=0)
    z_all = raw[:, :2]           # (STEPS, 2) [x, y]

    # ------------------------------------------------------------------
    # 2. Measurements for the filter: position z and its covariance R.
    #    None means "no usable fix at this step" (the filter only predicts).
    # ------------------------------------------------------------------

    # List of z (positions) and R (position covariance)
    zs, Rs = [], []
    for i in range(STEPS):
        z, R = None, None
        if np.all(np.isfinite(z_all[i])):   # find_points succeeded
            floater_xy = floaters[i, :, :2]
            try:
                R = measurement_covariance(floater_xy, z_all[i], sigma_deg)
                z = z_all[i]
            except np.linalg.LinAlgError: # Bearing directions are parallel
                pass
        zs.append(z)
        Rs.append(R)


    # First fix precise enough to start the track: a fix close to the baseline
    # has a huge ellipse and must not be used as starting point

    # Choose a sufficiently precise point to start the tracking, to avoid a wrong filter initialization
    start = None
    for i in range(STEPS):
        # Look 
        if zs[i] is not None and np.sqrt(np.trace(Rs[i])) <= max_init_std:
            start = i
            break

    # TODO: In questo caso vorrei semplicemente fermare il filtro, non l'intero programma
    if start is None:
        raise ValueError("No position fix precise enough to start the filter "
                         "(increase max_init_std or check the bearings).")

    # State vector has 4 elements [x, vx, y, vy], measurement vector has 2 [x,y]
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # F: motion model matrix
    kf.F = np.array([[1, dt, 0, 0],
                     [0, 1,  0, 0],
                     [0, 0,  1, dt],
                     [0, 0,  0, 1]])

    # H: measurement model. The measurement is the position itself: H extracts [x,y] from state [x, vx, y, vy].
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0]], dtype=float)

    # Q: process noise covariance, the uncertainty added at every prediction
    #    because the vessel may accelerate or turn. It models a random
    #    acceleration of intensity a_max^2 * dt on each axis.
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
            # Gating: squared Mahalanobis distance of the fix from the prediction,
            # i.e. the size of the innovation y measured in units of its expected
            # uncertainty S (prediction uncertainty + measurement uncertainty)
            y = z - kf.H @ kf.x                   # innovation
            S = kf.H @ kf.P @ kf.H.T + R          # innovation covariance
            d2 = y @ np.linalg.solve(S, y)        # y^T S^-1 y

            if d2 <= gate:
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

        # (if z is None there is no measurement: the prediction is kept)
        x_f.append(kf.x.copy())
        P_f.append(kf.P.copy())

    x_f, P_f = np.array(x_f), np.array(P_f)

    # RTS smoother
    x_s, _, _, _ = kf.rts_smoother(x_f, P_f)

    # TODO: "steps before 'start' stay NaN". Should it be better to substitute them with raw data?
    filtered = np.full((STEPS, 3), np.nan)
    filtered[start:, :2] = x_f[:, [0, 2]]   # Filitered [x,y]
    filtered[:, 2] = raw[:, 2] # Depth filling

    smoothed = np.full((STEPS, 3), np.nan)
    smoothed[start:, :2] = x_s[:, [0, 2]]   # Smoothed [x,y]
    smoothed[:, 2] = raw[:, 2] # Depth filling

    velocity = np.full((STEPS, 2), np.nan)
    velocity[start:] = x_s[:, [1, 3]]  # Velocity components [vx, vy]

    return dict(raw=raw, filtered=filtered, smoothed=smoothed, velocity=velocity)