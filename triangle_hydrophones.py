import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from utils_filters import *
import numpy as np
from utils_filters import *
from utils import *

MAX_SAMPLE_DELAY = 25

d = 0.31 # Meters
c = 1500 # Meters/second
tau_thr = 5 * 10**(-7)

if __name__ == "__main__":
    precompute_bearing_angles(d)
    fs, data = wav.read('AudioFiles/0958.wav')

    # Secondi di inizio e fine lettura
    start = 0
    #end = 350
    end = len(data)

    start *= fs
    end *= fs 
    sig1 = data[start:end, 0]
    sig2 = data[start:end, 1]
    sig3 = data[start:end, 2]
    
    # Filtering (?)
    '''
    fc = 10
    fc2 = 44000
    N = 201
    sig1 = FIR_bandpass_filter(sig1,fc,fc2,fs,N)
    sig2 = FIR_bandpass_filter(sig2,fc,fc2,fs,N)
    sig3 = FIR_bandpass_filter(sig3,fc,fc2,fs,N)
    '''

    # Parametri finestra
    durata_finestra = 0.05 # Secondi
    campioni_finestra = int(durata_finestra * fs)
    print(f"FREQUENZA DI CAMPIONAMENTO: {fs}")
    print(f"FINESTRA: {durata_finestra*1000} ms - {campioni_finestra} samples")

    '''
    sample_delay_21, times = compute_sample_delay(sig2,sig1,fs,campioni_finestra)
    sample_delay_32, _ = compute_sample_delay(sig3,sig2,fs,campioni_finestra)
    sample_delay_31, _ = compute_sample_delay(sig3,sig1,fs,campioni_finestra)

    '''

    quality_threshold = 0.2
    sample_delay_21, times, tau_percentile_21  = compute_sample_delay_d_aware(sig2,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold)
    sample_delay_32, _, tau_percentile_32 = compute_sample_delay_d_aware(sig3,sig2,fs,campioni_finestra,d,quality_threshold=quality_threshold)
    sample_delay_31, _, tau_percentile_31 = compute_sample_delay_d_aware(sig3,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold)


    datasets_sample_delay = [sample_delay_21,sample_delay_32,sample_delay_31]
    datasets_label = ["tau21","tau32","tau31"]

    fig, axes = plt.subplots(1, 5, figsize=(10, 10), sharey=True)

    time_delay_21 = sample_delay_21 / fs
    time_delay_32 = sample_delay_32 / fs
    time_delay_31 = sample_delay_31 / fs

    # Bearing estimation and error calculation
    tau_fit_error = np.zeros(len(times))
    estimated_bearing = np.zeros(len(times))
    for i in range(len(estimated_bearing)):
        estimated_bearing[i],tau_fit_error[i] = find_bearing(time_delay_32[i],time_delay_21[i],time_delay_31[i])

    mask_tau = (tau_fit_error < tau_thr) # Mask to consider only tau estimation with low error

    # ------------------------------ PLOTTING ------------------------------
    # Plot delay (in number of samples) for each pair of hydrophones
    for ax, sampledelay, lbl in zip(axes, datasets_sample_delay, datasets_label):
        ax.scatter(sampledelay, times, color='blue', s=0.5)
        ax.invert_yaxis()
        ax.set_title(lbl)
        ax.set_xlim(-MAX_SAMPLE_DELAY,MAX_SAMPLE_DELAY)
        ax.set_xlabel(f'# Of samples', fontsize=10, color='blue')
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[0].set_ylabel('Time (s)', fontsize=10)

    # Plot Bearing angle
    estimated_bearing = (estimated_bearing + 30) % 360

    ax = axes[3]
    ax.scatter(estimated_bearing[mask_tau],times[mask_tau],color='red',s=0.5) #Masked
    #ax.scatter(estimated_bearing,times,color='red',s=0.5)
    ax.set_xlim(0, 360)
    ax.set_xlabel(f'Bearing (°)', fontsize=10, color='red')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Plot tau error
    ax = axes[4]
    ax.plot(tau_fit_error,times,color='orange',linewidth=0.2)
    ax.set_xlabel(f'Error', fontsize=10, color='orange')
    ax.axvline(x=tau_thr, linewidth=1, color='blue')
    #ax.set_xlim(0, (10**(-4)))
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    #sig2_cropped = sig2[0:5*fs]
    #wav.write("rumore.wav",96000,sig2_cropped.astype(np.int16))
    plt.show()