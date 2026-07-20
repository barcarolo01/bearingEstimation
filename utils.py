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

        
def compute_sample_delay_value(sig_A, sig_B, fs, campioni_finestra, d, c=1500, overlap=0.5, quality_threshold=0.1):
    step = int(campioni_finestra * (1 - overlap))
    n_finestre = 1 + (len(sig_A) - campioni_finestra) // step
    sample_delay = np.zeros(n_finestre)
    times        = np.arange(n_finestre) * step / fs

    # Range fisicamente possibile
    tau_max_samples = int(np.ceil(d / c * fs))+1

    i = 0
    for inizio in range(0, min(len(sig_A), len(sig_B)) - campioni_finestra, step):
        fine      = inizio + campioni_finestra
        finestra1 = sig_A[inizio:fine]
        finestra2 = sig_B[inizio:fine]
        cc     = gcc_phat(finestra1, finestra2)
        center = len(cc) // 2

        # Cerca SOLO nel range fisico +-tau_max_samples
        search = cc[center - tau_max_samples : center + tau_max_samples + 1]
        peak   = np.max(search)


        if peak >= quality_threshold:
            lag = np.argmax(search) - tau_max_samples  # in campioni, relativo a lag=0
        else:
            lag = np.nan   # finestra scartata
        sample_delay[i] = lag
        i += 1

    return sample_delay, times

def compute_sample_delay_array(sig_A, sig_B, fs, campioni_finestra, d, c=1500, overlap=0.5, quality_threshold=0.1):
    step = int(campioni_finestra * (1 - overlap))
    n_finestre = 1 + (len(sig_A) - campioni_finestra) // step
    sample_delay = np.zeros(n_finestre)
    times        = np.arange(n_finestre) * step / fs

    # Range fisicamente possibile
    tau_max_samples = int(np.ceil(d / c * fs)) + 3
    
    i = 0
    searches = []
    for inizio in range(0, min(len(sig_A), len(sig_B)) - campioni_finestra, step):
        fine      = inizio + campioni_finestra
        finestra1 = sig_A[inizio:fine]
        finestra2 = sig_B[inizio:fine]

        cc     = gcc_phat(finestra1, finestra2)
        #cc     = gcc_phat_lowpass(finestra1, finestra2, fc=25000)
        center = len(cc) // 2

        # Cerca SOLO nel range fisico ±tau_max_samples
        search = cc[center - tau_max_samples : center + tau_max_samples + 1]
        peak   = np.max(search)

        if peak >= quality_threshold:
            lag = np.argmax(search) - tau_max_samples  # in campioni, relativo a lag=0
        else:
            lag = np.nan   # finestra scartata
        sample_delay[i] = lag
        searches.append(search)
        i += 1

    searches_np = np.array(searches)
    return searches_np, sample_delay, times