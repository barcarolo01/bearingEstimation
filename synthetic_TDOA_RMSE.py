import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from utils_filters import *
import numpy as np
from utils_filters import *
from utils import *

desired_SNR = 2

MAX_SAMPLE_DELAY = 20

d = 0.3 # Meters
c = 1500 # Meters/second

if __name__ == "__main__":
    precompute_bearing_angles(d)
    fs, sig1 = wav.read('RX1.wav')
    _, sig2 = wav.read('RX2.wav')
    _, sig3 = wav.read('RX3.wav')
    
    # Parametri finestra
    durata_finestra = 0.05 # Secondi
    campioni_finestra = int(durata_finestra * fs)
    print(f"FREQUENZA DI CAMPIONAMENTO: {fs}")
    print(f"FINESTRA: {durata_finestra*1000} ms - {campioni_finestra} samples")
    
    quality_threshold = 0
    sample_delay_21, times, tau_percentile_21  = compute_sample_delay_d_aware(sig2,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold)
    sample_delay_32, _, tau_percentile_32 = compute_sample_delay_d_aware(sig3,sig2,fs,campioni_finestra,d,quality_threshold=quality_threshold)
    sample_delay_31, _, tau_percentile_31 = compute_sample_delay_d_aware(sig3,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold)


    datasets_sample_delay = [sample_delay_21,sample_delay_32,sample_delay_31]
    datasets_label = ["tau21","tau32","tau31"]

    time_delay_21 = sample_delay_21 / fs
    time_delay_32 = sample_delay_32 / fs
    time_delay_31 = sample_delay_31 / fs

    # Bearing estimation and error calculation
    tau_fit_error = np.zeros(len(times))
    estimated_bearing = np.zeros(len(times))
    for i in range(len(estimated_bearing)):
        estimated_bearing[i],tau_fit_error[i] = find_bearing(time_delay_32[i],time_delay_21[i],time_delay_31[i])


    # ========================================================================
    #for desired_SNR in range (3,-10,-1):
    desired_SNR = -3
    sig1_noisy = add_gaussian_noise(sig1,desired_SNR)
    sig2_noisy = add_gaussian_noise(sig2,desired_SNR)
    sig3_noisy = add_gaussian_noise(sig3,desired_SNR)

    sample_delay_21_noisy, times_noisy, _  = compute_sample_delay_d_aware(sig2_noisy,sig1_noisy,fs,campioni_finestra,d,quality_threshold=quality_threshold)
    sample_delay_32_noisy, _, _ = compute_sample_delay_d_aware(sig3_noisy,sig2_noisy,fs,campioni_finestra,d,quality_threshold=quality_threshold)
    sample_delay_31_noisy, _, _ = compute_sample_delay_d_aware(sig3_noisy,sig1_noisy,fs,campioni_finestra,d,quality_threshold=quality_threshold)
    time_delay_21_noisy = sample_delay_21_noisy / fs
    time_delay_32_noisy = sample_delay_32_noisy / fs
    time_delay_31_noisy = sample_delay_31_noisy / fs
    datasets_sample_delay_noisy = [sample_delay_21_noisy,sample_delay_32_noisy,sample_delay_31_noisy]

    estimated_bearing_noisy = np.zeros(len(times_noisy))
    for i in range(len(estimated_bearing_noisy)):
        estimated_bearing_noisy[i], _ = find_bearing(time_delay_32_noisy[i],time_delay_21_noisy[i],time_delay_31_noisy[i])
    
    
    diff = (estimated_bearing_noisy - estimated_bearing + 180) % 360 - 180
    RMSE_bearing = np.sqrt(np.mean(diff**2))
    print(f"SNR = {desired_SNR} dB \t RMSE = {RMSE_bearing:.2f}°")
    # ========================================================================

    # ------------------------------ PLOTTING ------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(10, 10), sharey=True)

    r1 = axes[0]
    for ax, sampledelay, lbl in zip(r1, datasets_sample_delay, datasets_label):
        ax.scatter(sampledelay, times, color='blue', s=0.5)
        ax.invert_yaxis()
        ax.set_title(lbl)
        ax.set_xlim(-MAX_SAMPLE_DELAY,MAX_SAMPLE_DELAY)
        ax.set_xlabel(f'# Of samples', fontsize=10, color='blue')
        ax.grid(True, linestyle='--', alpha=0.5)
    r1[1].set_ylabel('Time (s)', fontsize=10)

    r1[3].scatter(estimated_bearing,times,color='red',s=0.5) #Masked
    r1[3].set_xlim(0, 360)
    r1[3].set_xlabel(f'Bearing (°)', fontsize=10, color='red')
    r1[3].grid(True, linestyle='--', alpha=0.5)

    r2 = axes[1]
    for ax, sampledelay, lbl in zip(r2, datasets_sample_delay_noisy, datasets_label):
        ax.scatter(sampledelay, times, color='blue', s=0.5)
        ax.invert_yaxis()
        ax.set_title(lbl)
        ax.set_xlim(-MAX_SAMPLE_DELAY,MAX_SAMPLE_DELAY)
        ax.set_xlabel(f'# Of samples', fontsize=10, color='blue')
        ax.grid(True, linestyle='--', alpha=0.5)
    r2[1].set_ylabel('Time (s)', fontsize=10)

    r2[3].scatter(estimated_bearing_noisy,times,color='red',s=0.5) #Masked
    r2[3].set_xlim(0, 360)
    r2[3].set_xlabel(f'Bearing (°)', fontsize=10, color='red')
    r2[3].grid(True, linestyle='--', alpha=0.5)


    plt.tight_layout()
    plt.show()