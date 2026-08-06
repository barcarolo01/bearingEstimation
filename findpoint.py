import numpy as np
import math

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

    # Use approximate metric coordinates centred on the floaters position
    lat0 = np.mean(lats)
    lon0 = np.mean(lons)
    R = 6371000.0  # Earth radius in metres
    lat0_rad = np.deg2rad(lat0)

    # Converts lat/lon -> metres relative to the centre
    x0 = np.deg2rad(lons - lon0) * R * np.cos(lat0_rad)
    y0 = np.deg2rad(lats - lat0) * R

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
    """Distance in meters between two points (flat-earth approximation)"""
    meters_per_deg_lat = 111_319.9
    meters_per_deg_lon = 111_319.9 * np.cos(np.deg2rad((lat1 + lat2) / 2))
    dy = (lat2 - lat1) * meters_per_deg_lat
    dx = (lon2 - lon1) * meters_per_deg_lon
    return np.sqrt(dx**2 + dy**2)

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

    bearing = np.rad2deg(np.arctan2(dx, dy))  # arctan2(East, North) -> CW from North
    return float(bearing % 360)

def find_points(
    floaters: np.ndarray,
    bearings: np.ndarray,
    elevation_array: np.ndarray | None = None,
) -> np.ndarray:
    """
    For each of the N simulation steps, this method compute the point that minimzes the distance 
    among the M direction of arrival lines estimated by each of the M floaters

    Parameters
    ---------
    floaters : np.ndarray of shape (N, M, 2) or (N, M, 3)
        N simulation steps, each with M floaters. Coordinates must be in form [lat, lon] or [lat, lon, depth_m].
    bearings : np.ndarray of shape (M, N)
        Horizontal angles (degrees) for each of the M floaters and for each of the N simulation steps.
    elevation_array : np.ndarray of shape (M, N)
        Vertical angles (degrees) for each of the M floaters and for each of the N simulation steps.

    Returns
    -------
    positions : np.ndarray of shape (N, 3)
        A triple [latitude, longitude, depth_m] for each of the N simulation steps
    """

    floaters = np.asarray(floaters, dtype=float)
    bearings = np.asarray(bearings, dtype=float)

    if floaters.ndim != 3 or floaters.shape[2] not in (2, 3):
        raise ValueError(
            f"Floaters must have shape (N, M, 2) or (N, M, 3), while it has {floaters.shape}."
        )
    if bearings.ndim != 2:
        raise ValueError(
            f"Bearings must have shape (M, N), while it has {bearings.shape}."
        )

    n_simulations = floaters.shape[0]
    n_floaters = floaters.shape[1]


    if bearings.shape[0] != n_floaters or bearings.shape[1] != n_simulations:
        raise ValueError(
            f"Bearings must have shape ({n_floaters}, {n_simulations}), while it has {bearings.shape}."
        )
    if n_floaters < 2:
        raise ValueError("At least 2 floaters are needed")


    has_depth = floaters.shape[2] == 3
    use_elevation = has_depth and elevation_array is not None

    if use_elevation:
        elevation_array = np.asarray(elevation_array, dtype=float)
        if elevation_array.shape != bearings.shape:
            raise ValueError(
                f"Elevation_array must have the same shape of bearings {bearings.shape}, while it has {elevation_array.shape}."
            )

    
    positions = np.full((n_simulations, 3), np.nan)
    brgs = np.vectorize(math_to_bearing)(bearings)

    # Iteration over the N simulation steps
    for n in range(n_simulations):

        # Fetching data of the n-th simulation
        floaters_n = floaters[n]
        brg_n = brgs[:, n] # Shape (M,) (i.e.: one angle for each floater)
        
        lats = floaters_n[:, 0]
        lons = floaters_n[:, 1]
        floater_depths = floaters_n[:, 2] if has_depth else None

        # If only two floaters are present, the point of minimum distance is the intersection of the bearing lines
        if n_floaters == 2:
            lat_i, lon_i = _flat_earth_intersection(
                lats[0], lons[0], brg_n[0],
                lats[1], lons[1], brg_n[1],
            )
            if not np.isnan(lat_i):
                positions[n, 0] = lat_i
                positions[n, 1] = lon_i

        # If more than two floaters are used, estimate the point minimizing the distance between all bearing lines
        else:
            candidates = []
            for i in range(n_floaters):
                for j in range(i + 1, n_floaters):
                    lat_i, lon_i = _flat_earth_intersection(
                        lats[i], lons[i], brg_n[i],
                        lats[j], lons[j], brg_n[j],
                    )
                    if not np.isnan(lat_i):
                        candidates.append((lat_i, lon_i))

            opt = _least_squares_point_n(lats, lons, brg_n)

            if opt is None:
                if len(candidates) >= 2:
                    positions[n, 0] = float(np.mean([p[0] for p in candidates]))
                    positions[n, 1] = float(np.mean([p[1] for p in candidates]))
                elif len(candidates) == 1:
                    positions[n, 0] = candidates[0][0]
                    positions[n, 1] = candidates[0][1]
                else:
                    continue
            else:
                positions[n, 0] = opt[0]
                positions[n, 1] = opt[1]

        # Depth estimation
        if use_elevation and not np.isnan(positions[n, 0]):
            depth_estimates = []
            weights = []
            
            for i in range(n_floaters):
                el_deg = elevation_array[i, n] #i-th floater, n-th simulation step
                
                '''
                if abs(el_deg) > 85.0:
                    continue
                '''

                el_rad = np.deg2rad(el_deg)
                dist_h = _flat_dist_m(lats[i], lons[i], positions[n, 0], positions[n, 1])
                
                delta_z = dist_h * np.tan(el_rad)
                estimated_z = floater_depths[i] - delta_z
                
                depth_estimates.append(estimated_z)
                weights.append(np.cos(el_rad) ** 2)

            if depth_estimates:
                positions[n, 2] = float(np.average(depth_estimates, weights=weights))
            else:
                positions[n, 2] = float(np.mean(floater_depths))

    return positions