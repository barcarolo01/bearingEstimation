import os
import shutil
import numpy as np
import math
from scipy import stats
import scipy.io.wavfile as wav
from utils import *
from floater_geometry import *
from bearing_calculation import *

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
    overlap = 0.0
    sample_delay_21, times  = compute_sample_delay_value(sig2,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=overlap)
    sample_delay_32, _ = compute_sample_delay_value(sig3,sig2,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=overlap)
    sample_delay_31, _ = compute_sample_delay_value(sig3,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=overlap)

    time_delay_21 = sample_delay_21 / fs
    time_delay_32 = sample_delay_32 / fs
    time_delay_31 = sample_delay_31 / fs

    # Bearing estimation
    estimated_azimuth = np.zeros(len(times))
    for i in range(len(estimated_azimuth)):
        estimated_azimuth[i],_ = find_bearing_triangle(time_delay_32[i],time_delay_21[i],time_delay_31[i])

    estimated_azimuth = format_bearings(estimated_azimuth,durata_finestra,0.1)

    np.save(f"Synth/F{H_index}",estimated_azimuth)
    return estimated_azimuth

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
    estimated_azimuth = np.zeros(len(times))
    for i in range(len(estimated_azimuth)):
        estimated_azimuth[i] = find_bearing_square(time_delay_32[i],time_delay_21[i],time_delay_31[i],
                                                   time_delay_41[i],time_delay_42[i],time_delay_43[i])
        
    estimated_azimuth = format_bearings(estimated_azimuth,durata_finestra,0.1)
    np.save(f"Synth/F{H_index}",estimated_azimuth)
    return estimated_azimuth

def compute_bearing_angle_array_complete(H_index, DESIRED_SNR = 999):
    d = 0.228 / math.sqrt(2)
    precompute_bearing_angles_complete(d)

    fs, sig1 = wav.read(f'Synth/F{H_index}_H1.wav')
    _, sig2 = wav.read(f'Synth/F{H_index}_H2.wav')
    _, sig3 = wav.read(f'Synth/F{H_index}_H3.wav')
    _, sig4 = wav.read(f'Synth/F{H_index}_H4.wav')
    _, sig5 = wav.read(f'Synth/F{H_index}_H5.wav')

    
    if DESIRED_SNR < 999:
        sig1 = add_white_noise(sig1,snr_db=DESIRED_SNR,seed=1)
        sig2 = add_white_noise(sig2,snr_db=DESIRED_SNR,seed=2)
        sig3 = add_white_noise(sig3,snr_db=DESIRED_SNR,seed=3)
        sig4 = add_white_noise(sig4,snr_db=DESIRED_SNR,seed=4)
        sig5 = add_white_noise(sig5,snr_db=DESIRED_SNR,seed=5)
        
    
    # Parametri finestra
    durata_finestra = 0.05  # Secondi
    campioni_finestra = int(durata_finestra * fs)
    quality_threshold = 0.0
    
    # Ritardi tra H1–H4 (invariati)
    sample_delay_21, times = compute_sample_delay_value(sig2, sig1, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    sample_delay_32, _     = compute_sample_delay_value(sig3, sig2, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    sample_delay_31, _     = compute_sample_delay_value(sig3, sig1, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    sample_delay_41, _     = compute_sample_delay_value(sig4, sig1, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    sample_delay_42, _     = compute_sample_delay_value(sig4, sig2, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    sample_delay_43, _     = compute_sample_delay_value(sig4, sig3, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)

    # Ritardi con H5 → informazione sull'angolo verticale
    sample_delay_51, _ = compute_sample_delay_value(sig5, sig1, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    sample_delay_52, _ = compute_sample_delay_value(sig5, sig2, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    sample_delay_53, _ = compute_sample_delay_value(sig5, sig3, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    sample_delay_54, _ = compute_sample_delay_value(sig5, sig4, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)

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
        

    estimated_azimuth = format_bearings(estimated_azimuth,durata_finestra,0.1)
    estimated_elevation = format_bearings(estimated_elevation,durata_finestra,0.1)

    np.save(f"Synth/F{H_index}_azimuth",   estimated_azimuth)
    np.save(f"Synth/F{H_index}_elevation", estimated_elevation)
    return estimated_azimuth, estimated_elevation


def format_bearings(array,window_duration,perc_to_trim):
    final_length = int(1/window_duration)
    array += 360
    N_adjusted = (len(array) // final_length) * final_length
    dati_regolari = array[:N_adjusted]
    segment_matrix = dati_regolari.reshape(-1, final_length)
    array_trimmed = stats.trim_mean(segment_matrix, proportiontocut=perc_to_trim, axis=1)
    array = array_trimmed - 360
    return array
    


def clean_temporary_files():
    if os.path.isdir("TMP"):
        shutil.rmtree("TMP")

    for j in range(5):
        if os.path.isdir(f"HM_OUT_{j+1}"):
                shutil.rmtree(f"HM_OUT_{j+1}")