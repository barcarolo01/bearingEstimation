import os
import numpy as np
import math
import scipy.io.wavfile as wav
import numpy as np
from utils import *
from LUTs_computation import *
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
    d = 0.228 / math.sqrt(2)
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



def clean_temporary_files(dir_path):
    if os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(dir_path)