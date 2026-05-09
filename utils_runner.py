import numpy as np
import math
import scipy.io.wavfile as wav
from utils_filters import *
import numpy as np
from utils_filters import *
from utils import *

'''
This method receives as input the initial and final coordinates of the transmitter (vessel)
and an integer n_steps. It returns a list of coordinates equally spaced that connects 
initial coordinates to final ones on a straight line.
'''
def compute_TX_trajectory(Lat_TX_init,Lon_TX_init,Lat_TX_end,Lon_TX_end,n_steps):
    Lat_TXs = np.zeros(n_steps)
    Lon_TXs = np.zeros(n_steps)
    Lat_TXs[0] = Lat_TX_init
    Lon_TXs[0] = Lon_TX_init

    for i in range(1,n_steps):
        Lat_TXs[i] = Lat_TX_init + i*(Lat_TX_end - Lat_TX_init) / (n_steps - 1)
        Lon_TXs[i] = Lon_TX_init + i*(Lon_TX_end - Lon_TX_init) / (n_steps - 1)

    # Returns the array of coordinates in form of [n_steps,2]
    return np.asarray([Lat_TXs, Lon_TXs]).T


'''
This method receives as input the index of a floater.
It analyses the three tracks synthetized for each of the hydrophone of that floater,
applies a window-based analysis and estimates the bearing angle for each window.
An array of bearing angle is produces and saved as a file (numpy array).
'''
def compute_bearing_angle_array(H_index):
    d=0.3
    precompute_bearing_angles(d)
    fs, sig1 = wav.read(f'Synth/F{H_index}_H1.wav')
    _, sig2 = wav.read(f'Synth/F{H_index}_H2.wav')
    _, sig3 = wav.read(f'Synth/F{H_index}_H3.wav')

    # Parametri finestra
    durata_finestra = 0.05 # Secondi
    campioni_finestra = int(durata_finestra * fs)
    print(f"FREQUENZA DI CAMPIONAMENTO: {fs}")
    print(f"FINESTRA: {durata_finestra*1000} ms - {campioni_finestra} samples")
    
    quality_threshold = 0.0
    sample_delay_21, times  = compute_sample_delay_d_aware(sig2,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=0)
    sample_delay_32, _ = compute_sample_delay_d_aware(sig3,sig2,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=0)
    sample_delay_31, _ = compute_sample_delay_d_aware(sig3,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=0)

    time_delay_21 = sample_delay_21 / fs
    time_delay_32 = sample_delay_32 / fs
    time_delay_31 = sample_delay_31 / fs

    # Bearing estimation and error calculation
    tau_fit_error = np.zeros(len(times))
    estimated_bearing = np.zeros(len(times))
    for i in range(len(estimated_bearing)):
        estimated_bearing[i],tau_fit_error[i] = find_bearing(time_delay_32[i],time_delay_21[i],time_delay_31[i])

    np.save(f"Synth/F{H_index}",estimated_bearing)

    return estimated_bearing

def sposta(lat,lon,distanza,dir):
    # conversioni
    delta_lat = distanza / 111320
    delta_lon = distanza / (111320 * math.cos(math.radians(lat)))

    if dir == "N":
        new_lat = lat + delta_lat # NORD
        new_lon = lon
    if dir == "S":
        new_lat = lat - delta_lat # SUD
        new_lon = lon
    if dir == "E":
        new_lat = lat 
        new_lon = lon + delta_lon
    if dir == "O":
        new_lat = lat 
        new_lon = lon - delta_lon

    return new_lat, new_lon


def random_points_within_distance(lat, lon, N, D, seed=42):
    """
    Genera N punti casuali entro una distanza D da (lat, lon).

    Args:
        lat:  Latitudine del punto centrale (gradi)
        lon:  Longitudine del punto centrale (gradi)
        N:    Numero di punti da generare
        D:    Distanza massima in metri
        seed: Seed per riproducibilità

    Returns:
        np.ndarray di shape (N, 2) con colonne [lat, lon]
    """
    rng = np.random.default_rng(seed)

    # Converti D in gradi (approssimazione locale)
    # 1 grado di latitudine ≈ 111_320 m ovunque
    # 1 grado di longitudine ≈ 111_320 * cos(lat) m
    lat_rad = np.radians(lat)
    delta_lat_deg = D / 111_320
    delta_lon_deg = D / (111_320 * np.cos(lat_rad))

    # Campionamento uniforme in un disco tramite coordinate polari
    # r = sqrt(u) garantisce densità uniforme nell'area
    u = rng.uniform(0, 1, N)
    theta = rng.uniform(0, 2 * np.pi, N)

    r_lat = np.sqrt(u) * delta_lat_deg
    r_lon = np.sqrt(u) * delta_lon_deg

    lats = lat + r_lat * np.cos(theta)
    lons = lon + r_lon * np.sin(theta)

    return np.column_stack((lats, lons))