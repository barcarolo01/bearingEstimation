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
def compute_bearing_angle_array(F_index):
    d=0.3
    precompute_bearing_angles_triangle(d)
    
    fs, sig1 = wav.read(f'Synth/F{F_index}_H1.wav')
    _, sig2 = wav.read(f'Synth/F{F_index}_H2.wav')
    _, sig3 = wav.read(f'Synth/F{F_index}_H3.wav')

    
    # Parametri finestra
    durata_finestra = 0.05 # Secondi

    campioni_finestra = int(durata_finestra * fs)
    print(f"FREQUENZA DI CAMPIONAMENTO: {fs}")
    print(f"FINESTRA: {durata_finestra*1000} ms - {campioni_finestra} samples")
    
    quality_threshold = 0.0
    overlap = 0.0
    _, sample_delay_21, times  = compute_sample_delay_array(sig2,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=overlap)
    _, sample_delay_32, _ = compute_sample_delay_array(sig3,sig2,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=overlap)
    _, sample_delay_31, _ = compute_sample_delay_array(sig3,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=overlap)

    time_delay_21 = sample_delay_21 / fs
    time_delay_32 = sample_delay_32 / fs
    time_delay_31 = sample_delay_31 / fs

    # Bearing estimation
    estimated_azimuth = np.zeros(len(times))
    for i in range(len(estimated_azimuth)):
        estimated_azimuth[i],_ = find_bearing_triangle(time_delay_32[i],time_delay_21[i],time_delay_31[i])

    estimated_azimuth = format_bearings(estimated_azimuth,durata_finestra,0.1)

    np.save(f"Synth/F{F_index}",estimated_azimuth)
    return estimated_azimuth

def compute_bearing_angle_array_square(F_index):
    d=0.3
    precompute_bearing_angles_square(d)
    fs, sig1 = wav.read(f'Synth/F{F_index}_H1.wav')
    _, sig2 = wav.read(f'Synth/F{F_index}_H2.wav')
    _, sig3 = wav.read(f'Synth/F{F_index}_H3.wav')
    _, sig4 = wav.read(f'Synth/F{F_index}_H4.wav')

    # Parametri finestra
    durata_finestra = 0.05 # Secondi
    campioni_finestra = int(durata_finestra * fs)

    quality_threshold = 0.0
    _, sample_delay_21, times  = compute_sample_delay_array(sig2,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=0)
    _, sample_delay_32, _ = compute_sample_delay_array(sig3,sig2,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=0)
    _, sample_delay_31, _ = compute_sample_delay_array(sig3,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=0)
    _, sample_delay_41, _ = compute_sample_delay_array(sig4,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=0)
    _, sample_delay_42, _ = compute_sample_delay_array(sig4,sig2,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=0)
    _, sample_delay_43, _ = compute_sample_delay_array(sig4,sig3,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=0)

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
    np.save(f"Synth/F{F_index}",estimated_azimuth)
    return estimated_azimuth

def compute_bearing_angle_array_complete(wav_folder, timestamp, F_index, DESIRED_SNR = 999):
    d = 0.228 / math.sqrt(2)
    precompute_bearing_angles_complete(d)

    '''
    fs, sig1 = wav.read(os.path.join(wav_folder,f'F{F_index}_H1.wav'))
    _, sig2 = wav.read(os.path.join(wav_folder,f'F{F_index}_H2.wav'))
    _, sig3 = wav.read(os.path.join(wav_folder,f'F{F_index}_H3.wav'))
    _, sig4 = wav.read(os.path.join(wav_folder,f'F{F_index}_H4.wav'))
    _, sig5 = wav.read(os.path.join(wav_folder,f'F{F_index}_H5.wav'))
    '''

    sig1 = np.load(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H1.npy'))
    sig2 = np.load(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H2.npy'))
    sig3 = np.load(os.path.join(wav_folder,f'T{timestamp}_F{timestamp}_H3.npy'))
    sig4 = np.load(os.path.join(wav_folder,f'T{timestamp}_F{timestamp}_H4.npy'))
    sig5 = np.load(os.path.join(wav_folder,f'T{timestamp}_F{timestamp}_H5.npy'))
    fs = 96000

    
    durata_finestra = 0.05  # Seconds
    campioni_finestra = int(durata_finestra * fs)
    quality_threshold = 0.0
    
    # Delays between hydrophones H1–H4 
    _, sample_delay_21, times = compute_sample_delay_array(sig2, sig1, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_32, _     = compute_sample_delay_array(sig3, sig2, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_31, _     = compute_sample_delay_array(sig3, sig1, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_41, _     = compute_sample_delay_array(sig4, sig1, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_42, _     = compute_sample_delay_array(sig4, sig2, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_43, _     = compute_sample_delay_array(sig4, sig3, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)

    # Delays with respect to H5 
    _, sample_delay_51, _ = compute_sample_delay_array(sig5, sig1, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_52, _ = compute_sample_delay_array(sig5, sig2, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_53, _ = compute_sample_delay_array(sig5, sig3, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_54, _ = compute_sample_delay_array(sig5, sig4, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)

    # Samples to seconds
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

    np.save(f"Synth/F{F_index}_azimuth",   estimated_azimuth)
    np.save(f"Synth/F{F_index}_elevation", estimated_elevation)
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

















def compute_single_bearing_angle_complete(wav_folder, timestamp, F_index):
    d = 0.228 / math.sqrt(2)
    precompute_bearing_angles_complete(d)

    fs, sig1 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H1.wav'))
    _, sig2 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H2.wav'))
    _, sig3 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H3.wav'))
    _, sig4 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H4.wav'))
    _, sig5 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H5.wav'))
    
    durata_finestra = 0.05  # Seconds
    campioni_finestra = int(durata_finestra * fs)
    quality_threshold = 0.0
    
    # Delays between hydrophones H1–H4 
    _, sample_delay_21, times = compute_sample_delay_array(sig2, sig1, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_32, _     = compute_sample_delay_array(sig3, sig2, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_31, _     = compute_sample_delay_array(sig3, sig1, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_41, _     = compute_sample_delay_array(sig4, sig1, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_42, _     = compute_sample_delay_array(sig4, sig2, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_43, _     = compute_sample_delay_array(sig4, sig3, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)

    # Delays with respect to H5 
    _, sample_delay_51, _ = compute_sample_delay_array(sig5, sig1, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_52, _ = compute_sample_delay_array(sig5, sig2, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_53, _ = compute_sample_delay_array(sig5, sig3, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_54, _ = compute_sample_delay_array(sig5, sig4, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)

    # Samples to seconds
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

    # Estimation array: azimuth and elevation for each window
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

    # Obtaining a single value for bearing and elevation angle    
    bearing = circular_trim_mean(estimated_azimuth,0.2)
    elevation = circular_trim_mean(estimated_elevation,0.2)

    return bearing, elevation



def circular_trim_mean(angles, proportiontocut=0.1):
    """
    This function takes as input an array of angles (in degrees),
    sorts them maintaining the wrap-around between 0° and 360°,
    removes the percentage (proportiontocut) of values at the extremes,
    and finally return the mean value of the remaining ones
    """
    angles = np.sort(np.asarray(angles, dtype=float) % 360)
    # Gap between consecutive angles
    gaps = np.diff(np.append(angles, angles[0] + 360))
    i = np.argmax(gaps)
    start = (i + 1) % len(angles)
    unwrapped = np.concatenate([angles[start:], angles[:start] + 360])
    mean_angle = stats.trim_mean(unwrapped, proportiontocut) % 360
    return mean_angle 