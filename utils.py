import numpy as np
from gcc_phat import *
from utils_filters import *
import numpy as np
from precompute_LUTs import *
        
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

        # Cerca SOLO nel range fisico ±tau_max_samples
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