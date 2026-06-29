from findpoint import *
from utils_runner import *

def sposta(lat, lon, distanza, dir):
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


def generate_grid_of_samples(lat_orig, lon_orig, W, N):
    """
    Generates a grid of coordinates (Lat, Lon) centered on the specified origin.
    
    Input:
    - lat_orig, lon_orig: Coordinates of the origin.
    - W: Maximum displacement in North, East and West direction
    - N: Number of samples along North, East and West direction
    
    Output:
    - Numpy array of shape (2N * N, 2) containing the generated pairs (Lat, Lon).
    """
    # Earth radius in meters
    R_earth = 6378137.0
    
    step = W / (N - 1)

    # Usa il passo per creare l'asse X simmetrico in modo che abbia pixel quadrati
    spostamento_y = np.linspace(0, W, N)
    spostamento_x = np.arange(-(N-1), N) * step  # Generates 2N-1 points from -W to +W
    
    delta_lat = (spostamento_y / R_earth) * (180.0 / np.pi)
    fattore_lon = R_earth * np.cos(np.radians(lat_orig))
    delta_lon = (spostamento_x / fattore_lon) * (180.0 / np.pi)

    lat_griglia = lat_orig + delta_lat
    lon_griglia = lon_orig + delta_lon
    
    # Meshgrid generates 2D matrices (N nows x 2N cols)
    LON, LAT = np.meshgrid(lon_griglia, lat_griglia)
    coordinate_coppie = np.column_stack((LAT.ravel(), LON.ravel()))
    
    return coordinate_coppie


def compute_TX_circle_trajectory(
    Lat_center: float,
    Lon_center: float,
    constant_depth: float,
    start_deg: float,  # Angolo di inizio da NORD (0-360)
    end_deg: float,    # Angolo di fine da NORD (0-360)
    n_steps: int,
    radius_m: float,
    clockwise: bool = True, # Se True va in senso orario, se False in senso antiorario
) -> np.ndarray:
    """
    Calcola i punti lungo un arco di circonferenza tra start_deg e end_deg.
    Gli angoli sono misurati a partire da NORD in senso orario.
    
    Ritorna un numpy array di forma (n_steps, 3) con colonne [Lat, Lon, Depth].
    """
    # Raggio medio della Terra in metri
    R_EARTH = 6371000.0
    
    # Normalizziamo gli angoli di input nel range [0, 360) per sicurezza
    start_deg = start_deg % 360.0
    end_deg = end_deg % 360.0
    
    if clockwise:
        # In senso orario: l'angolo cresce. 
        # Se l'angolo finale è minore di quello iniziale, abbiamo scavalcato il Nord (0°)
        if end_deg < start_deg:
            actual_end_deg = end_deg + 360.0
        else:
            actual_end_deg = end_deg
    else:
        # In senso antiorario: l'angolo decresce.
        # Se l'angolo finale è maggiore di quello iniziale, abbiamo scavalcato il Nord al contrario
        if end_deg > start_deg:
            actual_end_deg = end_deg - 360.0
        else:
            actual_end_deg = end_deg

    # Generiamo gli angoli orari intermedi (lineari tra inizio e fine modificata)
    clock_angles_deg = np.linspace(start_deg, actual_end_deg, n_steps)
    
    # Conversione da angoli bussola (orari, Nord=0) ad angoli trigonometrici (antiorari, Est=0)
    trig_angles_deg = 90.0 - clock_angles_deg
    trig_angles_rad = np.radians(trig_angles_deg)
    
    # Calcolo delle distorsioni geometriche per la proiezione piatta locale
    lat_center_rad = np.radians(Lat_center)
    delta_lat_deg = (radius_m / R_EARTH) * (180.0 / np.pi)
    delta_lon_deg = delta_lat_deg / np.cos(lat_center_rad)
    
    # Calcolo coordinate assolute
    d_lat = delta_lat_deg * np.sin(trig_angles_rad)
    d_lon = delta_lon_deg * np.cos(trig_angles_rad)
    
    lats = Lat_center + d_lat
    lons = Lon_center + d_lon
    depths = np.full(n_steps, constant_depth)
    
    return np.column_stack((lats, lons, depths))
def random_points_within_distance_recursive(lat, lon, constant_depth, N, max_distance, seed=42):
    rng = np.random.default_rng(seed)

    points = []
    current_lat, current_lon = lat, lon

    for _ in range(N):
        lat_rad = np.radians(current_lat)
        delta_lat_deg = max_distance / 111_320
        delta_lon_deg = max_distance / (111_320 * np.cos(lat_rad))

        u = rng.uniform(0, 1)
        theta = rng.uniform(0, 2 * np.pi)

        new_lat = current_lat + np.sqrt(u) * delta_lat_deg * np.cos(theta)
        new_lon = current_lon + np.sqrt(u) * delta_lon_deg * np.sin(theta)

        points.append((new_lat, new_lon, constant_depth))
        current_lat, current_lon = new_lat, new_lon

    return np.array(points)
