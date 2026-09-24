import os
import shutil
import numpy as np
from digitalshadow.utils.gcc_phat import *
import math
from scipy import stats
import scipy.io.wavfile as wav
from digitalshadow.utils.utils import *
from digitalshadow.devices.floater_geometry import *
from digitalshadow.utils.bearing_calculation import *

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
        estimated_azimuth[i] = find_bearing_triangle(time_delay_32[i],time_delay_21[i],time_delay_31[i])

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






def compute_single_bearing_angle_triangle(wav_folder, timestamp, F_index, SNR_desired=10000, seed = 0):
    d = 0.3
    precompute_bearing_angles_triangle(d)

    fs, sig1 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H1.wav'))
    _, sig2 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H2.wav'))
    _, sig3 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H3.wav'))

    durata_finestra = 0.05  # Seconds
    campioni_finestra = int(durata_finestra * fs)
    quality_threshold = 0.0

    if SNR_desired < 999:
        sig1 = add_white_noise(sig1,SNR_desired,seed=seed+1)
        sig2 = add_white_noise(sig2,SNR_desired,seed=seed+2)
        sig3 = add_white_noise(sig3,SNR_desired,seed=seed+3)

    # Delays between hydrophones H1–H4 
    _, sample_delay_21, times = compute_sample_delay_array(sig2, sig1, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_32, _     = compute_sample_delay_array(sig3, sig2, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_31, _     = compute_sample_delay_array(sig3, sig1, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)

    # Samples to seconds
    time_delay_21 = sample_delay_21 / fs
    time_delay_32 = sample_delay_32 / fs
    time_delay_31 = sample_delay_31 / fs

    # Estimation array: azimuth and elevation for each window
    estimated_azimuth   = np.zeros(len(times))
    estimated_elevation = np.zeros(len(times))

    for i in range(len(times)):
        az = find_bearing_triangle(time_delay_32[i], time_delay_21[i], time_delay_31[i])
        estimated_azimuth[i]   = az
        estimated_elevation[i] = 0

    # Obtaining a single value for bearing and elevation angle    
    bearing = circular_trim_mean(estimated_azimuth,0.2)
    elevation = 0

    return bearing, elevation









def compute_single_bearing_angle_square(wav_folder, timestamp, F_index, SNR_desired=10000, seed = 0):
    d = 0.228 / math.sqrt(2)
    precompute_bearing_angles_complete(d)

    fs, sig1 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H1.wav'))
    _, sig2 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H2.wav'))
    _, sig3 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H3.wav'))
    _, sig4 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H4.wav'))

    durata_finestra = 0.05  # Seconds
    campioni_finestra = int(durata_finestra * fs)
    quality_threshold = 0.0

    if SNR_desired < 999:
        sig1 = add_white_noise(sig1,SNR_desired,seed=seed+1)
        sig2 = add_white_noise(sig2,SNR_desired,seed=seed+2)
        sig3 = add_white_noise(sig3,SNR_desired,seed=seed+3)
        sig4 = add_white_noise(sig4,SNR_desired,seed=seed+4)

    # Delays between hydrophones H1–H4 
    _, sample_delay_21, times = compute_sample_delay_array(sig2, sig1, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_32, _     = compute_sample_delay_array(sig3, sig2, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_31, _     = compute_sample_delay_array(sig3, sig1, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_41, _     = compute_sample_delay_array(sig4, sig1, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_42, _     = compute_sample_delay_array(sig4, sig2, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_43, _     = compute_sample_delay_array(sig4, sig3, fs, campioni_finestra, d*3, quality_threshold=quality_threshold, overlap=0)

    # Samples to seconds
    time_delay_21 = sample_delay_21 / fs
    time_delay_32 = sample_delay_32 / fs
    time_delay_31 = sample_delay_31 / fs
    time_delay_41 = sample_delay_41 / fs
    time_delay_42 = sample_delay_42 / fs
    time_delay_43 = sample_delay_43 / fs

    # Estimation array: azimuth and elevation for each window
    estimated_azimuth   = np.zeros(len(times))
    estimated_elevation = np.zeros(len(times))

    for i in range(len(times)):
        az = find_bearing_square(
            time_delay_32[i], time_delay_21[i], time_delay_31[i],
            time_delay_41[i], time_delay_42[i], time_delay_43[i],
        )
        estimated_azimuth[i]   = az
        estimated_elevation[i] = 0

    # Obtaining a single value for bearing and elevation angle    
    bearing = circular_trim_mean(estimated_azimuth,0.2)
    elevation = 0

    return bearing, elevation







def compute_single_bearing_angle_complete(wav_folder, timestamp, F_index, SNR_desired=10000, seed = 0):
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

    if SNR_desired < 999:
        sig1 = add_white_noise(sig1,SNR_desired,seed=seed+1)
        sig2 = add_white_noise(sig2,SNR_desired,seed=seed+2)
        sig3 = add_white_noise(sig3,SNR_desired,seed=seed+3)
        sig4 = add_white_noise(sig4,SNR_desired,seed=seed+4)
        sig5 = add_white_noise(sig5,SNR_desired,seed=seed+5)
    
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


def check_snr(sig, noisy_sig):
    """
    Check the SNR between the original singal 'sig' and a noisy version of it 'noisy_sig'
    """
    sig_float = sig.astype(np.float64)
    noisy_float = noisy_sig.astype(np.float64)

    noise = noisy_float - sig_float

    sig_power = np.mean(sig_float ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return np.inf

    snr_db = 10 * np.log10(sig_power / noise_power)
    return snr_db

def add_white_noise(sig, snr_db, seed=None):
    rng = np.random.default_rng(seed)

    sig_float = sig.astype(np.float64)
    sig_power = np.mean(sig_float ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear


    noise = rng.normal(0, np.sqrt(noise_power), size=sig_float.shape)

    noisy_sig = sig_float + noise

    if np.issubdtype(sig.dtype, np.integer):
        info = np.iinfo(sig.dtype)
        noisy_sig = np.clip(noisy_sig, info.min, info.max)
        noisy_sig = noisy_sig.astype(sig.dtype)

    return noisy_sig

'''
Estimates the delay between two signals using the GCC-PHAT
Inputs:
    sig_A: first signal
    sig_B: second signal (should have the same length of the first)
    fs = sampling frequency of the signals (should be the same for both tracks)
    d = distance between the two hydrophones
    c = speed of sound
    overlap = overlap degree between the windows (between 0 and 1)
    quality_threshold = delay values above this value will be discarded and replaced with NaN

Outputs:
    - searches_np = matrix of shape (N,M)
        N = number of window analyzed
        M = number of physically possible delay values considered per window
    - samples_delay = array of length N, containing the lag (in samples) of the GCC-PHAT peak in each of the N windows 
    - times = array of length N, assigning a timestamp (in seconds) to each analyzed window
'''
def compute_sample_delay_array(sig_A, sig_B, fs, samples_per_window, d, c=1500, overlap=0.5, quality_threshold=0.1):
    step = int(samples_per_window * (1 - overlap))
    n_of_windows = 1 + (len(sig_A) - samples_per_window) // step
    times = np.arange(n_of_windows) * step / fs

    # Physically possible range
    tau_max_samples = int(np.ceil(d / c * fs)) + 5
    

    searches = []
    sample_delay = []
    for start in range(0, min(len(sig_A), len(sig_B)) - samples_per_window + 1, step):
        win1 = sig_A[start:start+samples_per_window]
        win2 = sig_B[start:start+samples_per_window]

        cc     = gcc_phat(win1, win2)
        #cc     = gcc_phat_lowpass(win1, win2, fc=25000)

        center = len(cc) // 2

        # Search only in the physically possible range (+-tau_max_samples)
        search = cc[center - tau_max_samples : center + tau_max_samples + 1]
        k    = np.argmax(search)
        peak = search[k]

        if peak >= quality_threshold:   # Quality check
            delta = 0.0
            if 0 < k < len(search) - 1:
                y0, y1, y2 = search[k - 1], search[k], search[k + 1]
                den = y0 - 2 * y1 + y2
                if den != 0:
                    delta = 0.5 * (y0 - y2) / den
            lag = (k - tau_max_samples) + delta
        else:
            lag = np.nan    # Low quality: the delay values is discarded



        sample_delay.append(lag)
        searches.append(search)

    searches_np = np.array(searches) 
    sample_delay = np.asarray(sample_delay)
    return searches_np, sample_delay, times