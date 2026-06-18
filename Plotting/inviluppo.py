from matplotlib.pylab import rfft, rfftfreq
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, welch, decimate
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from utils_filters import *
import numpy as np
from utils_filters import *
from utils import *

def compute_welch(envelope_dec,env_fs):
    nperseg = min(len(envelope_dec), int(env_fs / 0.1))
    nperseg = max(nperseg, 1024)
    noverlap = nperseg // 2
    nfft = nperseg * 2
    freqs, psd = welch(envelope_dec, fs=env_fs, nperseg=nperseg,
                       noverlap=noverlap, nfft=nfft,
                       window='hann', scaling='density')
    
    return freqs,psd

def decimate_envelope(envelope, sr, target_sf=1000):
    """
    Decimazione dell'inviluppo a target_sf (Hz).
    """
    
    factor = int(sr // target_sf)
    env_dec = decimate(envelope, factor,ftype='fir', zero_phase=True)
    new_fs = len(env_dec) / (len(envelope) / sr)
    return env_dec.astype(np.float32), new_fs

if __name__ == "__main__":
    fs, signal = wav.read('AudioFiles/0958_crop.wav')    
    sig0 = signal[:,1]
    f1 = 200
    f2 = 40000

    # Envelope and decimation
    envelope = np.abs(hilbert(sig0))
    '''
    hilbert(...) restituisce il segnale ANALITICO (stesso array dei tempi dell'argomento),
    in cui le frequenze positive vengono sfasate di 90° e quelle negative di -90°.
    Calcolando il modulo si calola l'inviluppo del segnale.
    '''

    envelope_dec, env_fs = decimate_envelope(envelope, fs, target_sf=1000)

    '''
    Downsampling necessario per il passaggio successivo (welch)
    '''

    # Passa alto
    hpf = 4.0 #Hz
    #envelope_dec = FIR_bandpass_filter(envelope_dec,hpf,env_fs//2,env_fs)

    # 1. FFT
    N = len(envelope_dec)
    fft_dec = rfft(envelope_dec)
    freqs_fft = rfftfreq(N, d=1/env_fs)
    mask_100 = freqs_fft <= 100

    # 2. Welch
    freqs,psd = compute_welch(envelope_dec,env_fs)
    mask = freqs <= 100


    # ================== PLOTTING ==================
    fig, axes = plt.subplots(3, 1, figsize=(16, 8))

    ax = axes[0]
    ax.plot(envelope_dec, color='red', linewidth=0.1)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title("Envelope (downsampled)",size=10)

    ax = axes[1]
    ax.plot(freqs_fft[mask_100], abs(fft_dec[mask_100]), color='blue', linewidth=0.1)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel("Hz")
    ax.set_title("FFT(envelope_dec)",size=10)
    
    ax = axes[2]
    ax.plot(freqs[mask], psd[mask], color='green', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel("Hz")
    ax.set_title("Welch \"DEMON\" Spectrum",size=10)

    plt.show()