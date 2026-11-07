import numpy as np
import math

def math_to_bearing(math_angle_deg):
    """
    Converts an angle expressed in the mathematical convention
    (0° = EAST, angles increasing in anti-clockwise sense) into an angle measured
    in geographical convention (0° = NORTH, angles increasing in clockwise sense)
    """
    return (90.0 - math_angle_deg) % 360.0

def two_line_intersection(x1, y1, brg1, x2, y2, brg2):
    """
    Computes the intersection between two lines in the local plane.

    Input 
        x1, y1 : coordinates of the starting point of the first line
        brg1   : first line direction (Geographical convention)
        x2, y2 : coordinates of the starting point of the second line
        brg2   : second line direction (Geographical convention)

    Output 
        Triangulated point (x,y), or (nan, nan) if no intersection point is found.
    """

    # Compute the direction vector in the plane (x=East, z=North)
    r = math.radians(brg1)
    dx1, dy1 = math.sin(r), math.cos(r)
    r = math.radians(brg2)
    dx2, dy2 = math.sin(r), math.cos(r)

    # Intersection of two parametric lines (Cramer)
    denom = dx1 * dy2 - dy1 * dx2 

    # Lines are parallel or coincident: no intersection can be estimated
    if abs(denom) < 1e-12:      
        return math.nan, math.nan

    # Parametric form of the linear system
    t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denom
    s = ((x2 - x1) * dy1 - (y2 - y1) * dx1) / denom

    # Line intersection is behind one of the two half-lines
    if t < 0 or s < 0:
        return math.nan, math.nan
    
    # Local coordinates of the intersection point
    x_out = x1 + t * dx1
    y_out = y1 + t * dy1

    return x_out, y_out

def _least_squares_point_n(xs, ys, brgs, weights):
    """
    Finds the point that minimizes the weighted sum of
    squared distances from the N half-lines in the local plane.

    Input:
        xs, ys: floater positions, shape (N,)
        brgs: bearing of each floater (Geographical convention), shape (N,)
        weights: weight of each floater, shape (N,). If None, uniform weights are used.

    Output
        (x_opt, y_opt): local coordinates of the optimal point, or None if the system is singular.
    """

    if weights is None:
        weights = np.ones(len(xs))

    # Unit directions of the lines in (dx=East, dz=North) coordinates
    brgs_rad = np.deg2rad(brgs)
    dx = np.sin(brgs_rad)  # East component
    dz = np.cos(brgs_rad)  # North component

    A = np.zeros((2, 2))
    b = np.zeros(2)
    for i in range(len(xs)):
        d = np.array([dx[i], dz[i]])
        P_orth = np.eye(2) - np.outer(d, d)
        h = np.array([xs[i], ys[i]])
        A += weights[i] * P_orth
        b += weights[i] * (P_orth @ h)

    try:
        p = np.linalg.solve(A, b)

    except np.linalg.LinAlgError:
        return None

    if not np.all(np.isfinite(p)):
        return None

    x_opt = p[0]
    y_opt = p[1]
    return x_opt, y_opt

def compute_flat_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Distance in meters between two points in the local plane"""
    dx = x2 - x1
    dz = y2 - y1
    return np.sqrt(dx**2 + dz**2)

def triangulate(floaters: np.ndarray, bearings: np.ndarray, elevation_array: np.ndarray = None, weights: np.ndarray = None):
    """
    For each of the N simulation steps, this method compute the point that minimzes the weighted RMSE
    distance with respect to the M direction of arrival lines estimated by each of the M floaters

    Input (I = simulation steps, N = floaters.):
        floaters: floater positions, shape (I, N, 2) or (I, N, 3). Coordinates must be in form [x, z] or [x, z, depth_m]
        bearings: DoA estimations, shape (N, I)
        elevation_array: elevation estimations, shape (N, I)
        weights: reliability of each DoA estimated, shape (N, I). These weights are normalized to sum up to 1

    Output
        positions: target triangulated positions, shape (I, 3)
    """

    # Check "floaters" and "bearings" arrays
    if floaters.ndim != 3 or floaters.shape[2] not in (2, 3):
        raise ValueError(f"Floaters must have shape (N, M, 2) or (N, M, 3), while it has {floaters.shape}.")
    if bearings.ndim != 2:
        raise ValueError(f"Bearings must have shape (M, N), while it has {bearings.shape}.")

    n_simulations = floaters.shape[0]
    n_floaters = floaters.shape[1]

    if bearings.shape[0] != n_floaters or bearings.shape[1] != n_simulations:
        raise ValueError(f"Bearings must have shape ({n_floaters}, {n_simulations}), while it has {bearings.shape}.")
    if n_floaters < 2:
        raise ValueError("At least 2 floaters are needed")

    has_depth = floaters.shape[2] == 3 # Boolean flag
    use_elevation = has_depth and (elevation_array is not None)

    if use_elevation:
        elevation_array = np.asarray(elevation_array, dtype=float)
        if elevation_array.shape != bearings.shape:
            raise ValueError(f"Elevation_array must have the same shape of bearings {bearings.shape}, while it has {elevation_array.shape}.")

    # Weights: check and normalization (sum = 1 for each simulation step)
    if weights is None:
        weights = np.ones_like(bearings)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != bearings.shape:
            raise ValueError(f"Weights must have the same shape of bearings {bearings.shape}, while it has {weights.shape}.")

    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    weight_sums = weights.sum(axis=0, keepdims=True)
    weights = np.where(weight_sums > 0, weights / np.where(weight_sums > 0, weight_sums, 1.0), 1.0 / n_floaters)

    positions = np.full((n_simulations, 3), np.nan)
    brgs = math_to_bearing(bearings)

    # Iteration over the N simulation steps
    for i in range(n_simulations):
        # Fetching data of the n-th simulation
        floaters_i = floaters[i]
        brg_n = brgs[:, i] # Shape (N,) (i.e.: one angle for each floater)
        w_n = weights[:, i] # Shape (N,) (i.e.: one weight for each floater)
        xs = floaters_i[:, 0]
        ys = floaters_i[:, 1]
        floater_depths = floaters_i[:, 2] if has_depth else None

        # If only two floaters are present, the point of minimum distance is the intersection of the bearing lines
        if n_floaters == 2:
            x_i, y_i = two_line_intersection(xs[0], ys[0], brg_n[0],xs[1], ys[1], brg_n[1],)
            if (not np.isnan(x_i) and not np.isnan(y_i)):
                positions[i, 0] = x_i
                positions[i, 1] = y_i

        # If more than two floaters are used, estimate the point minimizing the weighted distance between all bearing lines
        else:
            candidates = []
            candidate_weights = []

            # Compute the two-line intersection between each pair of floaters and keep each solution as a candidate
            for n in range(n_floaters):
                for j in range(n + 1, n_floaters):
                    x_i, y_i = two_line_intersection(xs[n], ys[n], brg_n[n],xs[j], ys[j], brg_n[j])
                    if (not np.isnan(x_i) and not np.isnan(y_i)):
                        candidates.append((x_i, y_i))
                        candidate_weights.append(w_n[i] * w_n[j])

             
            opt = _least_squares_point_n(xs, ys, brg_n, w_n)

            if opt is None:
                if len(candidates) >= 2:
                    cw = np.asarray(candidate_weights)
                    if cw.sum() <= 0:
                        cw = np.ones_like(cw)
                    positions[i, 0] = float(np.average([p[0] for p in candidates], weights=cw))
                    positions[i, 1] = float(np.average([p[1] for p in candidates], weights=cw))
                elif len(candidates) == 1:
                    positions[i, 0] = candidates[0][0]
                    positions[i, 1] = candidates[0][1]
                else:
                    continue
            else:
                positions[i, 0] = opt[0]
                positions[i, 1] = opt[1]

        # Depth estimation
        if use_elevation and not np.isnan(positions[i, 0]):
            depth_estimates = []
            depth_weights = []
            
            for n in range(n_floaters):
                el_deg = elevation_array[n, i] #i-th floater, n-th simulation step
                
                el_rad = np.deg2rad(el_deg)
                dist_h = compute_flat_distance(xs[n], ys[n], positions[i, 0], positions[i, 1])
                
                delta_z = dist_h * np.tan(el_rad)
                estimated_z = floater_depths[n] - delta_z
                
                depth_estimates.append(estimated_z)
                depth_weights.append(w_n[n] * np.cos(el_rad) ** 2)

            if depth_estimates and np.sum(depth_weights) > 0:
                positions[i, 2] = float(np.average(depth_estimates, weights=depth_weights))
            else:
                positions[i, 2] = float(np.mean(floater_depths))

    return positions