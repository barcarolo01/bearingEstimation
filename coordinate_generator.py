from point_estimation import *
from utils_runner import *

import numpy as np

def local_to_geo(Center_coordinates, local_point):
    local_point = np.asarray(local_point)

    Lat_center, Lon_center = Center_coordinates[:2]

    R = 6371000.0

    # Local coordinates
    x = local_point[..., 0]
    y = local_point[..., 1]

    # Conversion
    Lat = Lat_center + (y / R) * (180 / np.pi)
    Lon = Lon_center + (x / (R * np.cos(np.radians(Lat_center)))) * (180 / np.pi)

    # Maintain the depth if local_point contains 3D points
    if local_point.shape[-1] == 3:
        depth = local_point[..., 2]
        return np.stack((Lat, Lon, depth), axis=-1)
    elif local_point.shape[-1] == 2:
        return np.stack((Lat, Lon), axis=-1)
    else:
        raise ValueError("local_point should contain a collection of either 2D or 3D points")


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

def sposta(center, distanza, dir):
    lat, lon = center[0],center[1]
    
    # Converti i gradi in radianti
    angolo_rad = math.radians(dir)
    
    # Componenti Nord/Sud ed Est/Ovest usando seno e coseno
    delta_lat = distanza * math.cos(angolo_rad) / 111320
    delta_lon = distanza * math.sin(angolo_rad) / (111320 * math.cos(math.radians(lat)))
    
    new_lat = lat + delta_lat
    new_lon = lon + delta_lon
    
    return new_lat, new_lon

def calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calcola la distanza in metri tra due punti geografici (lat, lon),
    assumendo la Terra piatta.

    Args:
        point1: array numpy [lat, lon] in gradi decimali
        point2: array numpy [lat, lon] in gradi decimali

    Returns:
        Distanza in metri
    """
    R = 6_371_000  # raggio medio della Terra in metri

    lat1, lon1 = np.radians(point1)
    lat2, lon2 = np.radians(point2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    lat_mid = (lat1 + lat2) / 2

    x = dlon * np.cos(lat_mid)  # correzione per la convergenza dei meridiani
    y = dlat

    return R * np.sqrt(x**2 + y**2)

def generate_TX_trajectory(Lat_TX_init,Lon_TX_init,Lat_TX_end,Lon_TX_end,constant_depth,n_steps):
    '''
    This method receives as input the initial and final coordinates of the transmitter (vessel)
    and an integer n_steps. It returns a list of coordinates equally spaced that connects 
    initial coordinates to final ones on a straight line.
    '''
    Lat_TXs = np.zeros(n_steps)
    Lon_TXs = np.zeros(n_steps)
    depths = np.ones(n_steps)*constant_depth
    Lat_TXs[0] = Lat_TX_init
    Lon_TXs[0] = Lon_TX_init

    for i in range(1,n_steps):
        Lat_TXs[i] = Lat_TX_init + i*(Lat_TX_end - Lat_TX_init) / (n_steps - 1)
        Lon_TXs[i] = Lon_TX_init + i*(Lon_TX_end - Lon_TX_init) / (n_steps - 1)

    # Returns the array of coordinates in form of [n_steps,2]
    return np.asarray([Lat_TXs, Lon_TXs, depths]).T

def generate_grid_of_samples(lat_orig, lon_orig, depth_constant, W, N):
    """
    Generates a grid of coordinates (Lat, Lon) centered on the specified origin.

    Input:
    - lat_orig, lon_orig: Coordinates of the origin.
    - W: Maximum displacement in each of the four directions (N, S, E, W)
    - N: Number of samples per direction (origin included)

    Output:
    - Numpy array of shape ((2N-1)^2, 2) containing the generated pairs (Lat, Lon).
    """
    # Earth radius in meters
    R_earth = 6378137.0

    # 2N-1 punti equispaziati da -W a +W su entrambi gli assi -> pixel quadrati
    spostamento_y = np.linspace(-W, W, 2 * N - 1)
    spostamento_x = np.linspace(-W, W, 2 * N - 1)

    delta_lat = (spostamento_y / R_earth) * (180.0 / np.pi)
    fattore_lon = R_earth * np.cos(np.radians(lat_orig))
    delta_lon = (spostamento_x / fattore_lon) * (180.0 / np.pi)

    lat_griglia = lat_orig + delta_lat
    lon_griglia = lon_orig + delta_lon

    # Meshgrid generates 2D matrices ((2N-1) rows x (2N-1) cols)
    LON, LAT = np.meshgrid(lon_griglia, lat_griglia)
    coordinate_coppie = np.column_stack((LAT.ravel(), LON.ravel()))

    return coordinate_coppie

def compute_TX_circle_trajectory(
    Center,
    constant_depth: float,
    start_deg: float,  
    end_deg: float,  
    n_steps: int,
    radius_m: float,
    clockwise: bool = True):
    """
    This function returns a set of equally spaced points on a circumference
    between the specified angles (measured from North, clockwise)
    """
    Lat_center, Lon_center = Center[0], Center[1]

    R_EARTH = 6371000.0

    # Angle normalization
    start_deg = start_deg % 360.0
    end_deg = end_deg % 360.0
    
    if clockwise:
        if end_deg < start_deg:
            actual_end_deg = end_deg + 360.0
        else:
            actual_end_deg = end_deg
    else:
        if end_deg > start_deg:
            actual_end_deg = end_deg - 360.0
        else:
            actual_end_deg = end_deg

    # Generate a series of n_steps equally-spaced angles
    clock_angles_deg = np.linspace(start_deg, actual_end_deg, n_steps)

    # Angle conversion from Geographic (North = 0, clockwise) to mathematical convention (East = 0, anticlockwise)
    trig_angles_deg = 90.0 - clock_angles_deg
    trig_angles_rad = np.radians(trig_angles_deg)
    
    
    lat_center_rad = np.radians(Lat_center)
    delta_lat_deg = (radius_m / R_EARTH) * (180.0 / np.pi)
    delta_lon_deg = delta_lat_deg / np.cos(lat_center_rad)
    d_lat = delta_lat_deg * np.sin(trig_angles_rad)
    d_lon = delta_lon_deg * np.cos(trig_angles_rad)
    
    lats = Lat_center + d_lat
    lons = Lon_center + d_lon
    depths = np.full(n_steps, constant_depth)
    
    return np.column_stack((lats, lons, depths))