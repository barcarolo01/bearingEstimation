import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
from scipy.signal import butter, sosfiltfilt
from digitalshadow.utils.utils import *
from digitalshadow.devices.floater_geometry import *
from digitalshadow.utils.bearing_calculation import *


FONTSIZE = 16
MAX_SAMPLE_DELAY = 23
PLOT_COLORMAP = True
BUTTERWORTH_FILTER = False

d = 0.31 # Meters
c = 1500 # Meters/second

def filtro_passabanda(segnale, fs, f_bassa, f_alta, ordine=4):
    nyq = fs / 2.0 
    sos = butter(ordine, [f_bassa, f_alta], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(segnale, dtype=float))



if __name__ == "__main__":
    precompute_bearing_angles_triangle(d)

    '''
    # Reading 3-tracks file
    fs, data = wav.read('AudioFiles/0958_crop.wav')

    # Audio clipping: Start and end time (seconds)
    start = 0
    end = (len(data)//fs)

    sig1 = data[start*fs:end*fs, 0]
    sig2 = data[start*fs:end*fs, 1]
    sig3 = data[start*fs:end*fs, 2]
    '''

    fs, sig1 = wav.read('simulationGarda/SynthTracks/F1_H1.wav')
    _, sig2 = wav.read('simulationGarda/SynthTracks/F1_H2.wav')
    _, sig3 = wav.read('simulationGarda/SynthTracks/F1_H3.wav')


    # Signal pre-filtering: if used, spurious peaks are introduced in tau = 0
    if BUTTERWORTH_FILTER:
        sig1 = filtro_passabanda(sig1,fs,100,20000)
        sig2 = filtro_passabanda(sig2,fs,100,20000)
        sig3 = filtro_passabanda(sig3,fs,100,20000)    

    # Window parameters
    durata_finestra = 0.05 # Seconds
    overlap = 0.25
    quality_threshold = 0.0
    campioni_finestra = int(durata_finestra * fs)

    delay_arr21, sample_delay_21, times,_  = compute_sample_delay_array(sig2,sig1,fs,campioni_finestra,quality_threshold=quality_threshold,overlap=overlap)
    delay_arr32, sample_delay_32, _, _ = compute_sample_delay_array(sig3,sig2,fs,campioni_finestra,quality_threshold=quality_threshold,overlap=overlap)
    delay_arr31, sample_delay_31, _ , _= compute_sample_delay_array(sig3,sig1,fs,campioni_finestra,quality_threshold=quality_threshold,overlap=overlap)

    # Conversion: number of samples --> seconds
    time_delay_21 = sample_delay_21 / fs
    time_delay_32 = sample_delay_32 / fs
    time_delay_31 = sample_delay_31 / fs

    # Bearing estimation and error calculation
    estimated_bearing = np.zeros(len(times))
    for i in range(len(estimated_bearing)):
        estimated_bearing[i] = find_bearing_triangle(time_delay_32[i],time_delay_21[i],time_delay_31[i])

    # ==================== Plot Delays ====================
    datasets_sample_delay = [sample_delay_21,sample_delay_32,sample_delay_31]
    colormap_datasets = [delay_arr21,delay_arr32,delay_arr31]
    datasets_label = ["$\\tau_{21}$","$\\tau_{32}$","$\\tau_{31}$"]

    fig, axes = plt.subplots(1, 4, figsize=(12, 7), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.95, wspace=0.3)
    for i, (ax, sample_delay, values, lbl) in enumerate(zip(axes, datasets_sample_delay, colormap_datasets, datasets_label)):
        grid = np.array(values)
        
        # Plotting the colormap of GCC-PHAT values for each time
        if PLOT_COLORMAP:
            n_cols = grid.shape[1]
            tau_max = (n_cols - 1) // 2
            half = (tau_max + 0.5) * 1e6 / fs   # +0.5: i bordi dei pixel, così il lag 0 cade al centro
            extent = [-half, half, times[-1], times[0]]
            im = ax.imshow(
                grid,
                extent=extent,
                aspect='auto',
                cmap='viridis',
                interpolation='nearest',
            )
            pad = 0.0 * half                  
            ax.set_xlim(-half - pad, half + pad)
            if i == 0:
                ax.set_ylabel('Time (s)', fontsize=FONTSIZE)
                first_im = im

        # Plotting only the position GCC-PHAT maximum value
        else:
            ax.scatter(sample_delay * 1e6 / fs, times, s=5, c='steelblue')
            ax.set_xlim(-MAX_SAMPLE_DELAY*(1e6 / fs), MAX_SAMPLE_DELAY*(1e6 / fs))
            ax.set_ylim(times[-1], times[0])
            ax.set_ylabel('Time (s)', fontsize=FONTSIZE)

            
        ax.set_title(lbl,fontsize=FONTSIZE)
        ax.set_xlabel('Delay ($\mu$s)', fontsize=FONTSIZE)
        ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
        ax.grid(False)

    # Adding axis for "GCC-PHAT peak value" if the colormap option is TRUE
    if PLOT_COLORMAP:
        fig.canvas.draw()
        pos = axes[0].get_position()
        cbar_ax = fig.add_axes([
            pos.x0 - 0.1,
            pos.y0,
            0.02,    
            pos.height
        ])
        cbar = fig.colorbar(first_im, cax=cbar_ax)
        cbar.set_label('GCC-PATH peak value', fontsize=FONTSIZE)
        cbar_ax.tick_params(labelsize=FONTSIZE)
        cbar.ax.tick_params(labelsize=FONTSIZE)
        cbar_ax.yaxis.set_ticks_position('left')
        cbar_ax.yaxis.set_label_position('left')


    # ==================== Plot Bearing angle ====================
    #estimated_bearing = (360-estimated_bearing) % 360
    print(estimated_bearing[0])
    ax = axes[3]
    ax.scatter(estimated_bearing,times,color='red', s = 1) 
    ax.set_xlim(0, 360)
    ax.set_title("$\\theta$", fontsize=FONTSIZE)
    ax.set_xlabel(f'Degrees (°)', fontsize=FONTSIZE)
    ax.set_xticks([0, 180, 360])
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)

    plt.savefig("delay_bearing_estimation.png",dpi=900)
    plt.show()