import os
import shutil
from dotenv import load_dotenv
import numpy as np
from digitalshadow.utils.gcc_phat import *
import math
from scipy import stats
import scipy.io.wavfile as wav
from digitalshadow.utils.utils import *
from digitalshadow.devices.floater_geometry import *
from digitalshadow.utils.bearing_calculation import *

MAX_HYDROPHONE_DISTANCE = 1 # Meters

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

def compute_sample_delay_array(sig_A, sig_B, fs, samples_per_window, c=1500, overlap=0.5, quality_threshold=0.1):
    step = int(samples_per_window * (1 - overlap))
    n_of_windows = 1 + (len(sig_A) - samples_per_window) // step
    times = np.arange(n_of_windows) * step / fs

    # Physically possible range
    tau_max_samples = int(np.ceil(MAX_HYDROPHONE_DISTANCE / c * fs)) 
    
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

def compute_single_bearing_angle_triangle(track_folder, timestamp, F_index, seed = 0, SNR_desired = 999999):
    precompute_bearing_angles_triangle(0.3)

    fs, sig1 = wav.read(os.path.join(track_folder,f'T{timestamp}_F{F_index}_H1.wav'))
    _, sig2 = wav.read(os.path.join(track_folder,f'T{timestamp}_F{F_index}_H2.wav'))
    _, sig3 = wav.read(os.path.join(track_folder,f'T{timestamp}_F{F_index}_H3.wav'))

    if SNR_desired < 100:
        sig1 = add_white_noise(sig1,SNR_desired,seed=seed+1)
        sig2 = add_white_noise(sig2,SNR_desired,seed=seed+2)
        sig3 = add_white_noise(sig3,SNR_desired,seed=seed+3)

    durata_finestra = 0.05  # Seconds
    campioni_finestra = int(durata_finestra * fs)
    quality_threshold = 0.01
    test, sample_delay_21, times = compute_sample_delay_array(sig2, sig1, fs, campioni_finestra, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_32, _     = compute_sample_delay_array(sig3, sig2, fs, campioni_finestra, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_31, _     = compute_sample_delay_array(sig3, sig1, fs, campioni_finestra, quality_threshold=quality_threshold, overlap=0)

    #plt.figure()
    #plt.plot(sample_delay_21)
    #plt.show()

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

def compute_single_bearing_angle_square(wav_folder, timestamp, F_index, seed = 0):
    fs, sig1 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H1.wav'))
    _, sig2 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H2.wav'))
    _, sig3 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H3.wav'))
    _, sig4 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H4.wav'))

    durata_finestra = 0.05  # Seconds
    campioni_finestra = int(durata_finestra * fs)
    quality_threshold = 0.0

    # Delays between hydrophones H1–H4 
    _, sample_delay_21, times = compute_sample_delay_array(sig2, sig1, fs, campioni_finestra, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_32, _     = compute_sample_delay_array(sig3, sig2, fs, campioni_finestra, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_31, _     = compute_sample_delay_array(sig3, sig1, fs, campioni_finestra, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_41, _     = compute_sample_delay_array(sig4, sig1, fs, campioni_finestra, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_42, _     = compute_sample_delay_array(sig4, sig2, fs, campioni_finestra, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_43, _     = compute_sample_delay_array(sig4, sig3, fs, campioni_finestra, quality_threshold=quality_threshold, overlap=0)

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

def compute_single_bearing_angle_complete(wav_folder, timestamp, F_index):
    fs, sig1 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H1.wav'))
    _, sig2 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H2.wav'))
    _, sig3 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H3.wav'))
    _, sig4 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H4.wav'))
    _, sig5 = wav.read(os.path.join(wav_folder,f'T{timestamp}_F{F_index}_H5.wav'))
    
    durata_finestra = 0.05  # Seconds
    campioni_finestra = int(durata_finestra * fs)
    quality_threshold = 0.0

    
    # Delays between hydrophones H1–H4 
    _, sample_delay_21, times = compute_sample_delay_array(sig2, sig1, fs, campioni_finestra, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_32, _     = compute_sample_delay_array(sig3, sig2, fs, campioni_finestra, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_31, _     = compute_sample_delay_array(sig3, sig1, fs, campioni_finestra, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_41, _     = compute_sample_delay_array(sig4, sig1, fs, campioni_finestra, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_42, _     = compute_sample_delay_array(sig4, sig2, fs, campioni_finestra, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_43, _     = compute_sample_delay_array(sig4, sig3, fs, campioni_finestra, quality_threshold=quality_threshold, overlap=0)

    # Delays with respect to H5 
    _, sample_delay_51, _ = compute_sample_delay_array(sig5, sig1, fs, campioni_finestra, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_52, _ = compute_sample_delay_array(sig5, sig2, fs, campioni_finestra, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_53, _ = compute_sample_delay_array(sig5, sig3, fs, campioni_finestra, quality_threshold=quality_threshold, overlap=0)
    _, sample_delay_54, _ = compute_sample_delay_array(sig5, sig4, fs, campioni_finestra, quality_threshold=quality_threshold, overlap=0)

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

def clean_temporary_files(sim_name):
    for j in range(5):
        if os.path.isdir(f"HM_OUT_{sim_name}_{j+1}"):
                shutil.rmtree(f"HM_OUT_{sim_name}_{j+1}")