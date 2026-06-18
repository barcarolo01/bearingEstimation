import numpy as np

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