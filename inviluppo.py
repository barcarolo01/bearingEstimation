from matplotlib.pylab import fftfreq, rfft, rfftfreq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fft import fft
from scipy.signal import butter, filtfilt, hilbert, welch
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from utils_filters import *
import numpy as np
from utils_filters import *
from utils_2 import *

def decimate_envelope(envelope, sr, target_sr=1000):
    """
    Decimazione dell'inviluppo a target_sr Hz con anti-aliasing.
    Il DEMON spectrum ci interessa solo fino a ~300-500 Hz,
    quindi non serve lavorare a 96 kHz sull'inviluppo.
    """
    from scipy.signal import decimate
    factor = int(sr // target_sr)
    if factor <= 1:
        return envelope, sr
    # decimate applica internamente un filtro anti-aliasing
    env_dec = envelope.copy()
    # Decomposizione in passi da al più 13 per stabilità numerica
    while factor > 1:
        step = min(factor, 10)
        env_dec = decimate(env_dec, step, ftype='fir', zero_phase=True)
        factor = factor // step
    actual_sr = len(env_dec) / (len(envelope) / sr)
    return env_dec.astype(np.float32), actual_sr

if __name__ == "__main__":
    fs, data = wav.read('AudioFiles/0958.wav')
    sig0 = data[:,1]
    f1 = 200
    f2 = 47000

    # Bandpass filter
    filtered_data = bandpass_filter(sig0, f1, f2, fs)

    # Envelope and decimation
    envelope = np.abs(hilbert(filtered_data))
    '''
    hilbert(...) restituisce il segnale ANALITICO (stesso array dei tempi dell'argomento),
    in cui le frequenze positive vengono sfasate di 90° e quelle negative di -90°.
    Calcolando il modulo si calola l'inviluppo del segnale.
    '''


    envelope_dec, env_fs = decimate_envelope(envelope, fs, target_sr=1000)
    '''
    Downsampling necessario per il passaggio successivo (welch)
    '''

    # Passa alto
    hpf = 4.0 #Hz
    envelope_dec = highpass_filter(envelope_dec, hpf ,env_fs, 10)

    nperseg = min(len(envelope_dec), int(env_fs / 0.1))
    nperseg = max(nperseg, 1024)
    noverlap = nperseg // 2
    nfft = nperseg * 2
    freqs, psd = welch(envelope_dec, fs=env_fs, nperseg=nperseg,
                       noverlap=noverlap, nfft=nfft,
                       window='hann', scaling='density')
    

    N = len(envelope_dec)
    fft_dec = rfft(envelope_dec)
    freqs_fft = rfftfreq(N, d=1/env_fs)
    mask_100 = freqs_fft <= 100


    # ================== PLOTTING ==================
    fig, axes = plt.subplots(3, 1, figsize=(16, 8))

    ax = axes[0]
    ax.plot(envelope_dec, color='red', linewidth=0.1)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title("Envelope (downsampled)",size=10)

    ax = axes[1]
    
    ax.plot(freqs_fft[mask_100], abs(fft_dec[mask_100]), color='blue', linewidth=0.1)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title("FFT(envelope_dec)",size=10)
    
    ax = axes[2]
    mask = freqs <= 100
    ax.plot(freqs[mask], psd[mask], color='green', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title("Welch \"DEMON\" Spectrum",size=10)

    plt.show()