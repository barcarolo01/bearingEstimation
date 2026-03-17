import numpy as np
import matplotlib.pyplot as plt
from gcc_phat import *
import scipy.io.wavfile as wav
from utils_filters import *
from utils_2 import *


MAX_SAMPLE_DELAY = 25

if __name__ == "__main__":
    precompute_bearing_angles()
    #fs, data = wav.read('AudioFiles/0946.wav') #OK 0958
    fs, data = wav.read('AudioFiles/0958.wav')
    start = 0 # Secondo di inizio lettura

    start *= fs # Conversione (secondi) -> (# campioni)
    sig0 = data[start:, 0]
    sig1 = data[start:, 1]
    sig2 = data[start:, 2]
    
    # Filtering (?)
    '''
    sig0 = highpass_filter(sig0, 4000, fs)
    sig1 = highpass_filter(sig1, 4000, fs)
    sig2 = highpass_filter(sig2, 4000, fs)
    '''

    # Parametri finestra
    durata_finestra = 0.01 # Secondi
    campioni_finestra = int(durata_finestra * fs)
    print(f"FREQUENZA DI CAMPIONAMENTO: {fs}")
    print(f"FINESTRA: {durata_finestra*1000} ms - {campioni_finestra} samples")
    overlap = 0.5

    sample_delay_12, times = compute_sample_delay(sig0,sig1,fs,campioni_finestra,overlap)
    sample_delay_23, _ = compute_sample_delay(sig1,sig2,fs,campioni_finestra,overlap)
    sample_delay_31, _ = compute_sample_delay(sig2,sig0,fs,campioni_finestra,overlap)
    
    datasets_sample_delay = [sample_delay_12,sample_delay_12,sample_delay_31]
    datasets_label = ["tau01","tau12","tau20"]

    fig, axes = plt.subplots(1, 5, figsize=(10, 10), sharey=True)

    time_delay_12 = sample_delay_12 / fs
    time_delay_23 = sample_delay_23 / fs
    time_delay_31 = sample_delay_31 / fs

    # Bearing estimation and error calculation
    tau_fit_error = np.zeros(len(times))
    estimated_bearing = np.zeros(len(times))
    for i in range(len(estimated_bearing)):
        estimated_bearing[i],tau_fit_error[i] = find_bearing(time_delay_23[i],time_delay_12[i],time_delay_31[i])

    mask = tau_fit_error < tau_thr # Mask to consider only tau estimation with low error

    # ------------------------------ PLOTTING ------------------------------
    # Plot delay (in number of samples) for each pair of hydrophones
    for ax, sampledelay, lbl in zip(axes, datasets_sample_delay, datasets_label):
        ax.scatter(sampledelay, times, color='blue', linewidth=0.1)
        ax.invert_yaxis()
        ax.set_title(lbl)
        ax.set_xlim(-MAX_SAMPLE_DELAY,MAX_SAMPLE_DELAY)
        ax.set_xlabel(f'# Of samples', fontsize=10, color='blue')
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[0].set_ylabel('Time (s)', fontsize=10)

    # Plot Bearing angle
    estimated_bearing = (estimated_bearing + 30) % 360
    ax = axes[3]
    ax.scatter(estimated_bearing[mask],times[mask],color='red',linewidth=0.2)
    ax.set_xlim(0, 360)
    ax.set_xlabel(f'Bearing (°)', fontsize=10, color='red')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Plot tau error
    ax = axes[4]
    ax.plot(tau_fit_error,times,color='orange',linewidth=0.2)
    ax.set_xlabel(f'Error', fontsize=10, color='orange')
    ax.axvline(x=tau_thr, linewidth=1, color='blue')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()