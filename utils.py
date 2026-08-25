import numpy as np
from gcc_phat import *

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
        peak   = np.max(search)

        # Quality check
        if peak >= quality_threshold:
            lag = np.argmax(search) - tau_max_samples  # Number of samples (relatie to lag=0)
        else:
            lag = np.nan # Low quality: the delay values is discarded

        sample_delay.append(lag)
        searches.append(search)

    searches_np = np.array(searches) 
    sample_delay = np.asarray(sample_delay)
    return searches_np, sample_delay, times