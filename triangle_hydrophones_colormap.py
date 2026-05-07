import numpy as np
import matplotlib
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from utils_filters import *
import numpy as np
from utils_filters import *
from utils import *

MAX_SAMPLE_DELAY = 25

d = 0.31 # Meters
c = 1500 # Meters/second
tau_thr = 5 * 10**(-9)

if __name__ == "__main__":
    precompute_bearing_angles(d)
    
    
    fs, data = wav.read('AudioFiles/0958_crop.wav')
    # Secondi di inizio e fine lettura
    start = 0
    #end = 1
    end = len(data)//fs

    start *= fs
    end *= fs 
    sig1 = data[start:end, 0]
    sig2 = data[start:end, 1]
    sig3 = data[start:end, 2]

    '''
    # Reading synth tracks
    fs,sig1 = wav.read('Synth/H1_RX1.wav')
    _,sig2 = wav.read('Synth/H1_RX2.wav')
    _,sig3 = wav.read('Synth/H1_RX3.wav')
    '''
    
    #wav.write('sig1afterFilt.wav',fs,sig1)
    # ==================== Filtering (?) ====================
    fc = 10
    fc2 = 30000
    N = 201
    #sig1 = FIR_bandpass_filter(sig1,fc,fc2,fs,N)
    #sig2 = FIR_bandpass_filter(sig2,fc,fc2,fs,N)
    #sig3 = FIR_bandpass_filter(sig3,fc,fc2,fs,N)    

    #sig1 = FIR_lowpass_filter(sig1,fc2,fs,N)
    #sig2 = FIR_lowpass_filter(sig2,fc2,fs,N)
    #sig3 = FIR_lowpass_filter(sig3,fc2,fs,N)    

    #sig1 = lowpass_filter_fft(sig1,fs,fc2)
    #sig2 = lowpass_filter_fft(sig2,fs,fc2)
    #sig3 = lowpass_filter_fft(sig3,fs,fc2)
    # =======================================================
    wav.write('sig1afterFilt.wav',fs,sig1)

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

    quality_threshold = 0.0
    delay_arr21,sample_delay_21, times, tau_percentile_21  = compute_sample_delay_colormap(sig2,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold)
    delay_arr32,sample_delay_32, _, tau_percentile_32 = compute_sample_delay_colormap(sig3,sig2,fs,campioni_finestra,d,quality_threshold=quality_threshold)
    delay_arr31,sample_delay_31, _, tau_percentile_31 = compute_sample_delay_colormap(sig3,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold)


    colormap_datasets = [delay_arr21,delay_arr32,delay_arr31]
    datasets_sample_delay = [sample_delay_21,sample_delay_32,sample_delay_31]
    #datasets_label = ["tau21","tau32","tau31"]
    datasets_label = ["$\\tau_{21}$","$\\tau_{32}$","$\\tau_{31}$"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 6), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.95, wspace=0.3)
    

    time_delay_21 = sample_delay_21 / fs
    time_delay_32 = sample_delay_32 / fs
    time_delay_31 = sample_delay_31 / fs

    # Bearing estimation and error calculation
    tau_fit_error = np.zeros(len(times))
    estimated_bearing = np.zeros(len(times))
    for i in range(len(estimated_bearing)):
        estimated_bearing[i],tau_fit_error[i] = find_bearing(time_delay_32[i],time_delay_21[i],time_delay_31[i])

    mask_tau = (tau_fit_error < tau_thr) # Mask to consider only tau estimation with low error

    datasets_sample_delay = [sample_delay_21,sample_delay_32,sample_delay_31]
    colormap_datasets = [delay_arr21,delay_arr32,delay_arr31]
    # ------------------------------ PLOTTING ------------------------------
    for i, (ax, values, lbl) in enumerate(zip(axes, colormap_datasets, datasets_label)):
        grid = np.array(values)

        im = ax.imshow(
            grid,
            extent=[-MAX_SAMPLE_DELAY*(1e6 / fs), MAX_SAMPLE_DELAY*(1e6 / fs), times[-1], times[0]],
            aspect='auto',
            cmap='viridis',
            interpolation='nearest',
        )
        ax.set_title(lbl)
        ax.set_xlabel('Delay ($\mu$s)', fontsize=10)
        
        ax.grid(False)

        if i == 0:
            ax.set_ylabel('Time (s)', fontsize=10)
            first_im = im

    
    # Ricava la posizione del primo asse DOPO il layout
    fig.canvas.draw()  # forza il calcolo delle posizioni
    pos = axes[0].get_position()

    # Crea un asse dedicato alla colorbar a sinistra del primo plot
    cbar_ax = fig.add_axes([
        pos.x0 - 0.1,   # a sinistra del primo asse (aggiusta l'offset se necessario)
        pos.y0,
        0.02,             # larghezza della colorbar
        pos.height
    ])

    fig.colorbar(first_im, cax=cbar_ax, label='GCC_PATH peak value')
    cbar_ax.yaxis.set_ticks_position('left')
    cbar_ax.yaxis.set_label_position('left')



    # Plot Bearing angle
    #estimated_bearing = (estimated_bearing + 30) % 360

    ax = axes[3]
    ax.scatter(estimated_bearing[mask_tau],times[mask_tau],color='red',s=0.5) #Masked
    #ax.scatter(estimated_bearing,times,color='red',s=0.5)
    ax.set_xlim(0, 360)
    ax.set_title("$\\theta$")
    ax.set_xlabel(f'Degrees (°)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

    '''
    # Plot tau error
    ax = axes[4]
    ax.plot(tau_fit_error,times,color='orange',linewidth=0.2)
    ax.set_xlabel(f'Error', fontsize=10, color='orange')
    ax.axvline(x=tau_thr, linewidth=1, color='blue')
    #ax.set_xlim(0, (10**(-4)))
    ax.grid(True, linestyle='--', alpha=0.5)
    '''
    #plt.tight_layout()
    plt.savefig("delay_bearing_estimation.png")
    plt.show()