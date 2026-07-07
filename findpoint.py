import numpy as np
import math
from itertools import combinations

def math_to_bearing(math_angle_deg: float) -> float:
    """
    This function convert an angle expressed in the mathematical convention
    (0° = EAST, angles increasing in anti-clockwise sense) into an angle measured
    in geographical convention (0° = NORTH, angles increasing in clockwise sense)
    """
    return (90.0 - math_angle_deg) % 360.0

def _flat_earth_intersection(
    lat1: float, lon1: float, brg1: float,
    lat2: float, lon2: float, brg2: float,
) -> tuple[float, float]:
    """
    This function compute the intersection between two lines assuming the Earth to be flat.

    Parameters
    ---------
    lat1, lon1 : coordinates of the starting point of the first line
    brg1       : first line direction
    lat2, lon2 : coordinates of the starting point of the second line
    brg2       : second line direction

    Returns
    -------
    Coordintes (lat, lon) of the intersection point, or (nan, nan) if no intersection point is found.
    """
    # Longitude scaling factor in the local plane
    cos_lat = math.cos(math.radians((lat1 + lat2) / 2))

    # Compute the direction vector in the plane (x=East, y=North)
    r = math.radians(brg1)
    dx1, dy1 = math.sin(r), math.cos(r)
    r = math.radians(brg2)
    dx2, dy2 = math.sin(r), math.cos(r)


    # Coordinates in the local plane (degrees, with scaled longitude)
    x1, y1 = lon1 * cos_lat, lat1
    x2, y2 = lon2 * cos_lat, lat2

    # Intersection of two parametric lines:
    #   P1 + t * d1 = P2 + s * d2
    # Solved for t using Cramer's rule
    denom = dx1 * dy2 - dy1 * dx2 # Determinant

    if abs(denom) < 1e-12:      # Lines are parallel or coincident lines
        return math.nan, math.nan

    t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denom

    if t < 0:                   # Line intersection is behind half-line 1
        return math.nan, math.nan

    s = ((x2 - x1) * dy1 - (y2 - y1) * dx1) / denom
    if s < 0:                   # Line intersection is behind half-line 2
        return math.nan, math.nan

    # Geographic coordinates of the intersection point
    lon_out = (x1 + t * dx1) / cos_lat
    lat_out =  y1 + t * dy1

    return lat_out, lon_out

def _least_squares_point_n(
    lats: np.ndarray,
    lons: np.ndarray,
    brgs: np.ndarray,
) -> tuple[float, float] | None:
    """
    Finds the point that minimizes the sum of squared distances
    from the N geodetic half-lines (flat-earth approximation).

    Parameters
    ---------
    lats, lons : array (N,) of floater coordinates in degrees.
    brgs       : array (N,) of geographic bearings in degrees (clockwise from North).

    Returns
    -------
    (lat, lon) of the optimal point, or None if the system is singular.
    """
    # Unit directions of the lines in (dx=East, dy=North) coordinates
    brgs_rad = np.deg2rad(brgs)
    dx = np.sin(brgs_rad)  # East component
    dy = np.cos(brgs_rad)  # North component

    # Least squares system: minimize sum_i dist(P, line_i)^2
    # The distance from point P=(x,y) to the line passing through (x0,y0)
    # with direction (dx,dy) is: |(P - H) x d| = (dy*(x-x0) - dx*(y-y0))
    # Matrix A and vector b of the normal system A^T A p = A^T b
    # with orthogonal projection: (I - d d^T) P = (I - d d^T) H

    # Use approximate metric coordinates centred on the floaters position
    lat0 = np.mean(lats)
    lon0 = np.mean(lons)
    R = 6371000.0  # Earth radius in metres
    lat0_rad = np.deg2rad(lat0)

    # Converts lat/lon -> metres relative to the centre
    x0 = np.deg2rad(lons - lon0) * R * np.cos(lat0_rad)
    y0 = np.deg2rad(lats - lat0) * R

    # Orthogonal projectors: for each line i, (I - d_i d_i^T)
    # A x = b  =>  sum_i (I - d_i d_i^T) @ p = sum_i (I - d_i d_i^T) @ h_i
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for i in range(len(lats)):
        d = np.array([dx[i], dy[i]])
        P_orth = np.eye(2) - np.outer(d, d)
        h = np.array([x0[i], y0[i]])
        A += P_orth
        b += P_orth @ h

    try:
        p = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    if not np.all(np.isfinite(p)):
        return None

    # Converts metres -> degrees
    lon_opt = lon0 + np.rad2deg(p[0] / (R * np.cos(lat0_rad)))
    lat_opt = lat0 + np.rad2deg(p[1] / R)

    return float(lat_opt), float(lon_opt)

def _flat_dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in metres between two nearby points, flat-earth approximation."""
    meters_per_deg_lat = 111_319.9
    meters_per_deg_lon = 111_319.9 * np.cos(np.deg2rad((lat1 + lat2) / 2))
    dy = (lat2 - lat1) * meters_per_deg_lat
    dx = (lon2 - lon1) * meters_per_deg_lon
    return np.sqrt(dx**2 + dy**2)

def _residual_error(lats, lons, bearings_deg, lat_est, lon_est) -> float:
    total = 0.0
    n = 0
    for i in range(len(lats)):
        # Compute distance between i-th floater and estimated point
        d = _flat_dist_m(lats[i], lons[i], lat_est, lon_est)

        # Skip distances very close to zero
        if d < 1e-6:
            continue

        az = _flat_azimuth(lats[i], lons[i], lat_est, lon_est)
        delta_angle = np.deg2rad(az - bearings_deg[i])
        total += np.sin(delta_angle) ** 2  # dimensionless, independent of d
        n += 1
    return total / n if n > 0 else np.inf

def _flat_azimuth(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes the geographic bearing (Clockwise from North, in degrees [0, 360))
    from point 1 to point 2, flat-earth approximation.

    Parameters
    ---------
    lat1, lon1 : coordinates of the starting point (decimal degrees)
    lat2, lon2 : coordinates of the destination point (decimal degrees)

    Returns
    -------
    Bearing in degrees, CW from North, in the range [0, 360).
    """
    R = 6371000.0
    lat0_rad = np.deg2rad((lat1 + lat2) / 2)  # mean latitude for the correction

    dx = np.deg2rad(lon2 - lon1) * R * np.cos(lat0_rad)  # East component
    dy = np.deg2rad(lat2 - lat1) * R                      # North component

    bearing = np.rad2deg(np.arctan2(dx, dy))  # arctan2(East, North) → CW from North
    return float(bearing % 360)


def find_points_fixed(
    floaters: np.ndarray,
    bearings: np.ndarray,
    elevation_array: np.ndarray | None = None,
) -> np.ndarray:
    """
    For each bearing column, computes the point that minimizes the distance
    from the N geodetic half-lines emitted by the N floaters.
    If floaters has a third column (depth) and elevation_array is not None,
    also estimates the target depth.

    Parameters
    ---------
    floaters : np.ndarray of shape (N, 2) or (N, 3)
        Coordinates [lat, lon] or [lat, lon, depth_m] in decimal degrees (depth in metres,
        positive = below the surface).
    bearings : np.ndarray of shape (N, M)
        M sets of N angles (degrees), CCW from East convention.
        Each column m contains the bearings of the N floater for event m.
    elevation_array : np.ndarray of shape (N, M), optional
        Elevation angles in degrees for each floater and each event.
        Positive = above the horizon, negative = below.
        Ignored if floater does not contain depth.

    Returns
    -------
    positions : np.ndarray of shape (M, 3)
        Columns [latitude, longitude, depth_m]. Rows without a solution → [NaN, NaN, NaN].
        If depth is not estimated, the third column will remain NaN or contain a fallback.
    """
    floaters = np.asarray(floaters, dtype=float)
    bearings = np.asarray(bearings, dtype=float)

    if floaters.ndim != 2 or floaters.shape[1] not in (2, 3):
        raise ValueError(
            f"floaters must be in shape (N, 2) or (N, 3), while it has {floaters.shape}."
        )
    if bearings.ndim != 2:
        raise ValueError(
            f"bearings must be in shape (N, M), while it has {bearings.shape}."
        )

    n_floaters = floaters.shape[0]
    n_events = bearings.shape[1]

    if bearings.shape[0] != n_floaters:
        raise ValueError(f"bearings must have {n_floaters} rows (one for each floater), while it has {bearings.shape[0]}.")
    if n_floaters < 2:
        raise ValueError("At least 2 floaters must be provided")

    # ── Check if floater depth data is available
    has_depth = floaters.shape[1] == 3
    use_elevation = has_depth and elevation_array is not None

    lats = floaters[:, 0]
    lons = floaters[:, 1]
    floater_depths = floaters[:, 2] if has_depth else None

    if use_elevation:
        elevation_array = np.asarray(elevation_array, dtype=float)
        if elevation_array.shape != bearings.shape:
            raise ValueError(f"elevation_array must have the same shape of bearings {bearings.shape}, while it has {elevation_array.shape}.")

    # RESTITUISCE UN UNICO ARRAY (M, 3) COME IN ORIGINE
    positions = np.full((n_events, 3), np.nan)

    brgs = np.vectorize(math_to_bearing)(bearings)  # (N, M)
    
    for m in range(n_events):
        brg_m = brgs[:, m]  # (N,) bearings for event m

        # ── Estimate lat/lon ──
        if n_floaters == 2:
            lat_i, lon_i = _flat_earth_intersection(
                lats[0], lons[0], brg_m[0],
                lats[1], lons[1], brg_m[1],
            )
            if not np.isnan(lat_i):
                positions[m, 0] = lat_i
                positions[m, 1] = lon_i
        else:
            candidates = []
            for i in range(n_floaters):
                for j in range(i + 1, n_floaters):
                    lat_i, lon_i = _flat_earth_intersection(
                        lats[i], lons[i], brg_m[i],
                        lats[j], lons[j], brg_m[j],
                    )
                    if not np.isnan(lat_i):
                        candidates.append((lat_i, lon_i))

            opt = _least_squares_point_n(lats, lons, brg_m)

            if opt is None:
                if len(candidates) >= 2:
                    opt = (
                        float(np.mean([p[0] for p in candidates])),
                        float(np.mean([p[1] for p in candidates])),
                    )
                elif len(candidates) == 1:
                    opt = candidates[0]
                else:
                    continue

            positions[m, 0] = opt[0]
            positions[m, 1] = opt[1]

        # ── Depth estimation (with robust weighting and vertical protection) ──
        if use_elevation and not np.isnan(positions[m, 0]):
            depth_estimates = []
            weights = []
            
            for i in range(n_floaters):
                el_deg = elevation_array[i, m]
                
                # Protezione dall'esplosione della tangente ad angoli quasi verticali (>85°)
                if abs(el_deg) > 85.0:
                    continue
                
                el_rad = np.deg2rad(el_deg)
                dist_h = _flat_dist_m(lats[i], lons[i], positions[m, 0], positions[m, 1])
                
                delta_z = dist_h * np.tan(el_rad)
                estimated_z = floater_depths[i] - delta_z
                
                depth_estimates.append(estimated_z)
                weights.append(np.cos(el_rad) ** 2)

            if depth_estimates:
                positions[m, 2] = float(np.average(depth_estimates, weights=weights))
            else:
                # Fallback se tutti i floater avevano angoli troppo verticali
                positions[m, 2] = float(np.mean(floater_depths))

    return positions


def find_points(
    floaters: np.ndarray,
    bearings: np.ndarray,
    elevation_array: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each bearing column, computes the point that minimizes the distance
    from the N geodetic half-lines emitted by the N floaters.
    If floaters has a third column (depth) and elevation_array is not None,
    also estimates the target depth.

    Parameters
    ---------
    floaters : np.ndarray of shape (N, 2) or (N, 3)
        Coordinates [lat, lon] or [lat, lon, depth_m] in decimal degrees (depth in metres,
        positive = below the surface).
    bearings : np.ndarray of shape (N, M)
        M sets of N angles (degrees), CCW from East convention.
        Each column m contains the bearings of the N floater for event m.
    elevation_array : np.ndarray of shape (N, M), optional
        Elevation angles in degrees for each floater and each event.
        Positive = above the horizon, negative = below.
        Ignored if floater does not contain depth.

    Returns
    -------
    positions : np.ndarray of shape (M, 2)
        Columns [latitude, longitude]. Rows without a solution → [NaN, NaN].
    depths : np.ndarray of shape (M,)
        Estimated depth in metres (positive = below surface) if
        floater has 3 columns and elevation_array is not None,
        otherwise array of -999.
    """
    floaters = np.asarray(floaters, dtype=float)
    bearings    = np.asarray(bearings,    dtype=float)

    if floaters.ndim != 2 or floaters.shape[1] not in (2, 3):
        raise ValueError(
            f"floaters must be in shape (N, 2) or (N, 3), while it has {floaters.shape}."
        )
    if bearings.ndim != 2:
        raise ValueError(
            f"bearings must be in shape (N, M), while it has {bearings.shape}."
        )

    n_floaters  = floaters.shape[0]
    n_events = bearings.shape[1]

    if bearings.shape[0] != n_floaters:
        raise ValueError(f"bearings must ahve {n_floaters} rows (one for each floater), while it hase {bearings.shape[0]}.")
    if n_floaters < 2:
        raise ValueError("At least 2 floaters must be provided")

    # ── Check if floater depth data is available
    has_depth = floaters.shape[1] == 3
    use_elevation = has_depth and elevation_array is not None

    lats = floaters[:, 0]
    lons = floaters[:, 1]
    floater_depths = floaters[:, 2] if has_depth else None

    if use_elevation:
        elevation_array = np.asarray(elevation_array, dtype=float)
        if elevation_array.shape != bearings.shape:
            raise ValueError(f"elevation_array must have the same same shape of bearings {bearings.shape},while it has {elevation_array.shape}.")

    positions = np.full((n_events, 3), np.nan)

    brgs = np.vectorize(math_to_bearing)(bearings)  # (N, M)
    for m in range(n_events):
        brg_m = brgs[:, m]  # (N,) bearings for event m

        # ── Estimate lat/lon

        # If only 2 floaters are provided, then the optimal point is the intersection of bearing lines
        if n_floaters == 2:
            lat_i, lon_i = _flat_earth_intersection(
                lats[0], lons[0], brg_m[0],
                lats[1], lons[1], brg_m[1],
            )
            if not np.isnan(lat_i):
                positions[m, 0] = lat_i
                positions[m, 1] = lon_i
        else:
            candidates = []
            # Compute the intersection for each computer of bearing lines
            for i in range(n_floaters):
                for j in range(i + 1, n_floaters):
                    lat_i, lon_i = _flat_earth_intersection(
                        lats[i], lons[i], brg_m[i],
                        lats[j], lons[j], brg_m[j],
                    )
                    if not np.isnan(lat_i):
                        candidates.append((lat_i, lon_i))

            opt = _least_squares_point_n(lats, lons, brg_m)

            
            if opt is None:
                if len(candidates) >= 2:
                    opt = (
                        float(np.mean([p[0] for p in candidates])),
                        float(np.mean([p[1] for p in candidates])),
                    )
                elif len(candidates) == 1:
                    opt = candidates[0]
                else:
                    continue

            positions[m, 0] = opt[0]
            positions[m, 1] = opt[1]

        # ── Depth estimation (only if use_elevation and valid position) ───
        if use_elevation and not np.isnan(positions[m, 0]):
            depth_estimates = []
            for i in range(n_floaters):
                el_rad  = np.deg2rad(elevation_array[i, m])

                # Compute the 2D distance between the floater and the previously estiamted point
                dist_h  = _flat_dist_m(lats[i], lons[i], positions[m, 0], positions[m, 1])
                
                # Compute the difference of elevation
                delta_z = dist_h * np.tan(el_rad)

                depth_estimates.append(floater_depths[i] - delta_z)
            positions[m,2] = float(np.mean(depth_estimates))

    return positions


def find_points_weighted(
    floaters: np.ndarray,
    bearings: np.ndarray,
    elevation_array: np.ndarray | None = None,
) -> np.ndarray:
    """
    For each event, computes the estimated position as a weighted average
    of all combinations of K floaters (K from 3 to N), with weight 1/error.
    """
    floaters = np.asarray(floaters, dtype=float)
    bearings    = np.asarray(bearings,    dtype=float)

    n_floaters   = floaters.shape[0]
    n_events  = bearings.shape[1]
    has_depth = floaters.shape[1] == 3
    use_elevation = has_depth and elevation_array is not None

    best_positions = np.full((n_events, 3), np.nan)

    # Pre-compute all estimates for all combinations
    all_combos = []
    for k in range(3, n_floaters + 1):
        for indices in combinations(range(n_floaters), k):
            idx = list(indices)
            floater_subset   = floaters[idx, :]
            bearing_subset = bearings[idx, :]
            elev_sub    = elevation_array[idx, :] if use_elevation else None

            pos = find_points(floater_subset, bearing_subset, elev_sub)  # (n_events, 3)
            all_combos.append((idx, pos))

    for m in range(n_events):
        weights   = []
        lats_acc  = []
        lons_acc  = []
        depth_acc = []

        for idx, intersection in all_combos:
            pos_m = intersection[m]  # (3,)

            if np.isnan(pos_m[0]):
                continue

            brgs_m = np.vectorize(math_to_bearing)(bearings[idx, m])

            error = _residual_error(floaters[idx, 0], floaters[idx, 1],brgs_m,pos_m[0], pos_m[1])
            
            if error <= 0 or not np.isfinite(error):
                continue

            weights.append(1.0 / error)
            lats_acc.append(pos_m[0])
            lons_acc.append(pos_m[1])
            depth_acc.append(pos_m[2])

        if not weights:
            continue

        w = np.array(weights)
        w /= w.sum()

        best_positions[m, 0] = np.dot(w, lats_acc)
        best_positions[m, 1] = np.dot(w, lons_acc)
        if use_elevation:
            best_positions[m, 2] = np.dot(w, depth_acc)

    return best_positions