import os
from dotenv import load_dotenv
import numpy as np
import math
import scipy.io.wavfile as wav
from utils_filters import *
import numpy as np
from utils_filters import *
from utils import *
from precompute_LUTs import *
from bearing_calculation import *

'''
This method receives as input the initial and final coordinates of the transmitter (vessel)
and an integer n_steps. It returns a list of coordinates equally spaced that connects 
initial coordinates to final ones on a straight line.
'''
def compute_TX_trajectory(Lat_TX_init,Lon_TX_init,Lat_TX_end,Lon_TX_end,constant_depth,n_steps):
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


'''
This method receives as input the index of a floater.
It analyses the three tracks synthetized for each of the hydrophone of that floater,
applies a window-based analysis and estimates the bearing angle for each window.
An array of bearing angle is produces and saved as a file (numpy array).
'''
def compute_bearing_angle_array(H_index):
    d=0.3
    precompute_bearing_angles_triangle(d)
    
    fs, sig1 = wav.read(f'Synth/F{H_index}_H1.wav')
    _, sig2 = wav.read(f'Synth/F{H_index}_H2.wav')
    _, sig3 = wav.read(f'Synth/F{H_index}_H3.wav')

    # Parametri finestra
    durata_finestra = 0.05 # Secondi
    campioni_finestra = int(durata_finestra * fs)
    print(f"FREQUENZA DI CAMPIONAMENTO: {fs}")
    print(f"FINESTRA: {durata_finestra*1000} ms - {campioni_finestra} samples")
    
    quality_threshold = 0.0
    sample_delay_21, times  = compute_sample_delay_value(sig2,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=0)
    sample_delay_32, _ = compute_sample_delay_value(sig3,sig2,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=0)
    sample_delay_31, _ = compute_sample_delay_value(sig3,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=0)

    time_delay_21 = sample_delay_21 / fs
    time_delay_32 = sample_delay_32 / fs
    time_delay_31 = sample_delay_31 / fs

    # Bearing estimation
    estimated_bearing = np.zeros(len(times))
    for i in range(len(estimated_bearing)):
        estimated_bearing[i],_ = find_bearing_triangle(time_delay_32[i],time_delay_21[i],time_delay_31[i])

    np.save(f"Synth/F{H_index}",estimated_bearing)

    return estimated_bearing[:-1]

def compute_bearing_angle_array_square(H_index):
    d=0.3
    precompute_bearing_angles_square(d)
    fs, sig1 = wav.read(f'Synth/F{H_index}_H1.wav')
    _, sig2 = wav.read(f'Synth/F{H_index}_H2.wav')
    _, sig3 = wav.read(f'Synth/F{H_index}_H3.wav')
    _, sig4 = wav.read(f'Synth/F{H_index}_H4.wav')

    # Parametri finestra
    durata_finestra = 0.05 # Secondi
    campioni_finestra = int(durata_finestra * fs)
    #print(f"FREQUENZA DI CAMPIONAMENTO: {fs}")
    #print(f"FINESTRA: {durata_finestra*1000} ms - {campioni_finestra} samples")
    
    quality_threshold = 0.0
    sample_delay_21, times  = compute_sample_delay_value(sig2,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=0)
    sample_delay_32, _ = compute_sample_delay_value(sig3,sig2,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=0)
    sample_delay_31, _ = compute_sample_delay_value(sig3,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=0)
    sample_delay_41, _ = compute_sample_delay_value(sig4,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=0)
    sample_delay_42, _ = compute_sample_delay_value(sig4,sig2,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=0)
    sample_delay_43, _ = compute_sample_delay_value(sig4,sig3,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=0)

    time_delay_21 = sample_delay_21 / fs
    time_delay_32 = sample_delay_32 / fs
    time_delay_31 = sample_delay_31 / fs
    time_delay_41 = sample_delay_41 / fs
    time_delay_42 = sample_delay_42 / fs
    time_delay_43 = sample_delay_43 / fs

    # Bearing estimation
    estimated_bearing = np.zeros(len(times))
    for i in range(len(estimated_bearing)):
        estimated_bearing[i] = find_bearing_square(time_delay_32[i],time_delay_21[i],time_delay_31[i],
                                                   time_delay_41[i],time_delay_42[i],time_delay_43[i])

    np.save(f"Synth/F{H_index}",estimated_bearing)

    return estimated_bearing[:-1]

def compute_bearing_angle_array_complete(H_index):
    d = 0.3
    precompute_bearing_angles_complete(d)

    fs, sig1 = wav.read(f'Synth/F{H_index}_H1.wav')
    _, sig2 = wav.read(f'Synth/F{H_index}_H2.wav')
    _, sig3 = wav.read(f'Synth/F{H_index}_H3.wav')
    _, sig4 = wav.read(f'Synth/F{H_index}_H4.wav')
    _, sig5 = wav.read(f'Synth/F{H_index}_H5.wav')

    # Parametri finestra
    durata_finestra = 0.05  # Secondi
    campioni_finestra = int(durata_finestra * fs)

    quality_threshold = 0.0

    # Ritardi tra H1–H4 (invariati)
    sample_delay_21, times = compute_sample_delay_value(sig2, sig1, fs, campioni_finestra, d, quality_threshold=quality_threshold, overlap=0)
    sample_delay_32, _     = compute_sample_delay_value(sig3, sig2, fs, campioni_finestra, d, quality_threshold=quality_threshold, overlap=0)
    sample_delay_31, _     = compute_sample_delay_value(sig3, sig1, fs, campioni_finestra, d, quality_threshold=quality_threshold, overlap=0)
    sample_delay_41, _     = compute_sample_delay_value(sig4, sig1, fs, campioni_finestra, d, quality_threshold=quality_threshold, overlap=0)
    sample_delay_42, _     = compute_sample_delay_value(sig4, sig2, fs, campioni_finestra, d, quality_threshold=quality_threshold, overlap=0)
    sample_delay_43, _     = compute_sample_delay_value(sig4, sig3, fs, campioni_finestra, d, quality_threshold=quality_threshold, overlap=0)

    # Ritardi con H5 → informazione sull'angolo verticale
    sample_delay_51, _ = compute_sample_delay_value(sig5, sig1, fs, campioni_finestra, d*10, quality_threshold=quality_threshold, overlap=0)
    sample_delay_52, _ = compute_sample_delay_value(sig5, sig2, fs, campioni_finestra, d*10, quality_threshold=quality_threshold, overlap=0)
    sample_delay_53, _ = compute_sample_delay_value(sig5, sig3, fs, campioni_finestra, d*10, quality_threshold=quality_threshold, overlap=0)
    sample_delay_54, _ = compute_sample_delay_value(sig5, sig4, fs, campioni_finestra, d*10, quality_threshold=quality_threshold, overlap=0)

    '''
    if H_index == 1:
        print("TAU_4_1:")
        print(sample_delay_41)
        print("TAU_5_4:")
        print(sample_delay_54)
        print("TAU_5_3:")
        print(sample_delay_53)
        print("TAU_5_2:")
        print(sample_delay_52)
        print("TAU_5_1:")
        print(sample_delay_51)
    '''
    
    # Conversione in secondi
    time_delay_21 = sample_delay_21 / fs
    time_delay_32 = sample_delay_32 / fs
    time_delay_31 = sample_delay_31 / fs
    time_delay_41 = sample_delay_41 / fs
    time_delay_42 = sample_delay_42 / fs
    time_delay_43 = sample_delay_43 / fs
    time_delay_51 = sample_delay_51 / fs
    time_delay_52 = sample_delay_52 / fs
    time_delay_53 = sample_delay_53 / fs
    time_delay_54 = sample_delay_54 / fs

    # Stima azimuth + elevazione per ogni finestra
    estimated_azimuth   = np.zeros(len(times))
    estimated_elevation = np.zeros(len(times))

    for i in range(len(times)):
        az, el = find_bearing_complete(
            time_delay_32[i], time_delay_21[i], time_delay_31[i],
            time_delay_41[i], time_delay_42[i], time_delay_43[i],
            time_delay_51[i], time_delay_52[i], time_delay_53[i], time_delay_54[i]
        )
        estimated_azimuth[i]   = az
        estimated_elevation[i] = el

    np.save(f"Synth/F{H_index}_azimuth",   estimated_azimuth)
    np.save(f"Synth/F{H_index}_elevation", estimated_elevation)

    return estimated_azimuth[:-1], estimated_elevation[:-1]


def sposta(lat, lon, distanza, dir):
    # Converti i gradi in radianti
    angolo_rad = math.radians(dir)
    
    # Componenti Nord/Sud ed Est/Ovest usando seno e coseno
    delta_lat = distanza * math.cos(angolo_rad) / 111320
    delta_lon = distanza * math.sin(angolo_rad) / (111320 * math.cos(math.radians(lat)))
    
    new_lat = lat + delta_lat
    new_lon = lon + delta_lon
    
    return new_lat, new_lon

def random_points_within_distance(lat, lon, constant_depth, N, max_distance, seed=42):
    """
    Genera N punti casuali entro una distanza D da (lat, lon).

    Args:
        lat:  Latitudine del punto centrale (gradi)
        lon:  Longitudine del punto centrale (gradi)
        N:    Numero di punti da generare
        max_distance:    Distanza massima in metri
        seed: Seed per riproducibilità

    Returns:
        np.ndarray di shape (N, 2) con colonne [lat, lon]
    """
    rng = np.random.default_rng(seed)

    # Converti D in gradi (approssimazione locale)
    # 1 grado di latitudine ≈ 111_320 m ovunque
    # 1 grado di longitudine ≈ 111_320 * cos(lat) m
    lat_rad = np.radians(lat)
    delta_lat_deg = max_distance / 111_320
    delta_lon_deg = max_distance / (111_320 * np.cos(lat_rad))

    # Campionamento uniforme in un disco tramite coordinate polari
    # r = sqrt(u) garantisce densità uniforme nell'area
    u = rng.uniform(0, 1, N)
    theta = rng.uniform(0, 2 * np.pi, N)

    r_lat = np.sqrt(u) * delta_lat_deg
    r_lon = np.sqrt(u) * delta_lon_deg

    lats = lat + r_lat * np.cos(theta)
    lons = lon + r_lon * np.sin(theta)
    depths = np.ones(len(lons)) * constant_depth

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

def clean_temporary_files(dir_path):
    if os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(dir_path)