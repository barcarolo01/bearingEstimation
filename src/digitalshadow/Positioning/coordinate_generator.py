import math
import numpy as np

def local_to_geo(Center_coordinates, local_points):
    """
    Converts a list of cartesian points into a list of geographical coordinates

    Input:
        Center_coordinates: Geographical coordinates of center (Lat-Lon), shape (2,) or (3,)
        local_points: array of local points, shape(N,2) or (N,3)

    Output:
        coints_coordintes: array of geographical coordinates, shape (N,2) or (N,3)
    """

    local_points = np.asarray(local_points)

    # Center coordintaes
    Lat_center, Lon_center = Center_coordinates[:2]

    R_Earth = 6371000.0

    # Local coordinates
    x = local_points[..., 0]
    y = local_points[..., 1]

    # Local -> Coordinates conversion
    Lat = Lat_center + (y / R_Earth) * (180 / np.pi)
    Lon = Lon_center + (x / (R_Earth * np.cos(np.radians(Lat_center)))) * (180 / np.pi)

    # Maintain the depth if local_points contains 3D points
    if local_points.shape[-1] == 3:
        depth = local_points[..., 2]
        return np.stack((Lat, Lon, depth), axis=-1)
    elif local_points.shape[-1] == 2:
        return np.stack((Lat, Lon), axis=-1)
    else:
        raise ValueError("local_points should contain a collection of either 2D or 3D points")

def geo_to_local(Center_coordinates, geo_coordinates):
    """
    Converts a list of a geographical coordinates into a list of local points

    Input:
        Center_coordinates: Geographical coordinates of center (Lat-Lon), shape (2,) or (3,)
        geo_coordinates: array of local geographical coordinates, shape(N,2) or (N,3)

    Output:
        coints_coordintes: array of local points, shape (N,2) or (N,3)
    """

    # Extract latitudes and longitudes
    Lats = geo_coordinates[..., 0]
    Lons = geo_coordinates[..., 1]

    # Extract depth if available
    if geo_coordinates.shape[-1]:
        depths = geo_coordinates[..., 2]
    else:
        depths = np.full_like(x, np.nan)

    Lat_center, Lon_center = Center_coordinates[0], Center_coordinates[1]

    # Degrees -> radians conversion
    lats = np.radians(Lats)
    lons = np.radians(Lons)

    R_Earth = 6371000.0
    x = R_Earth * (lons - np.radians(Lon_center)) * np.cos(np.radians(Lat_center))
    y = R_Earth * (lats - np.radians(Lat_center))

    return np.stack((x, y, depths), axis=-1)

def compute_TX_circle_trajectory(constant_depth,start_deg,end_deg,n_steps,radius_m,clockwise=True):
    """
    Computes a set of equally spaced points on a circumference centered at (0, 0)

    Input:
        constant_depth: target constant depth
        start_deg, end_deg: start and end angle (in degrees) of the circumference arc
        n_steps: number of points to be generated
        radius_m: radius of the circumference arc
        clockwise: determines the sorting of the returned numpy array

    Output:
        arc_coordinates: numpy array of shape (N,3) [Lat-Lon-depth]
    """

    # Angle normalization
    start_deg = start_deg % 360.0
    end_deg = end_deg % 360.0

    if clockwise:
        actual_end_deg = end_deg + 360.0 if end_deg < start_deg else end_deg
    else:
        actual_end_deg = end_deg - 360.0 if end_deg > start_deg else end_deg

    # Generate a series of n_steps equally-spaced angles
    clock_angles_deg = np.linspace(start_deg, actual_end_deg, n_steps)

    # Conversion from geographic (North = 0, clockwise) to mathematical convention (East = 0, anticlockwise)
    trig_angles_rad = np.radians(90.0 - clock_angles_deg)

    # Cartesian coordinates relative to the circle center
    x = radius_m * np.cos(trig_angles_rad)   # East
    y = radius_m * np.sin(trig_angles_rad)   # North
    depths = np.full(n_steps, constant_depth)

    return np.column_stack((x, y, depths))