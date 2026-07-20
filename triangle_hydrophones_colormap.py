import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
from utils import *
from floater_geometry import *
from bearing_calculation import *

MAX_SAMPLE_DELAY = 25
PLOT_COLORMAP = True

d = 0.31 # Meters
c = 1500 # Meters/second


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
    fs, sig1 = wav.read('Synth/F1_H1.wav')
    _, sig2 = wav.read('Synth/F1_H2.wav')
    _, sig3 = wav.read('Synth/F1_H3.wav')


    # Window parameters
    durata_finestra = 0.05 # Seconds
    overlap = 0
    quality_threshold = 0.01
    campioni_finestra = int(durata_finestra * fs)

    delay_arr21,sample_delay_21, times  = compute_sample_delay_array(sig2,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=overlap)
    delay_arr32,sample_delay_32, _ = compute_sample_delay_array(sig3,sig2,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=overlap)
    delay_arr31,sample_delay_31, _ = compute_sample_delay_array(sig3,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold,overlap=overlap)
    
    # Conversion: number of samples --> seconds
    time_delay_21 = sample_delay_21 / fs
    time_delay_32 = sample_delay_32 / fs
    time_delay_31 = sample_delay_31 / fs

    # Bearing estimation and error calculation
    tau_fit_error = np.zeros(len(times))
    estimated_bearing = np.zeros(len(times))
    for i in range(len(estimated_bearing)):
        estimated_bearing[i],tau_fit_error[i] = find_bearing_triangle(time_delay_32[i],time_delay_21[i],time_delay_31[i])

    # ==================== Plot Delays ====================
    datasets_sample_delay = [sample_delay_21,sample_delay_32,sample_delay_31]
    colormap_datasets = [delay_arr21,delay_arr32,delay_arr31]
    datasets_label = ["$\\tau_{21}$","$\\tau_{32}$","$\\tau_{31}$"]

    fig, axes = plt.subplots(1, 4, figsize=(12, 6), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.95, wspace=0.3)
    for i, (ax, values, lbl) in enumerate(zip(axes, colormap_datasets, datasets_label)):
        grid = np.array(values)
        
        # Plotting the colormap of GCC-PHAT values for each time
        if PLOT_COLORMAP:
            im = ax.imshow(
                grid,
                extent=[-MAX_SAMPLE_DELAY*(1e6 / fs), MAX_SAMPLE_DELAY*(1e6 / fs), times[-1], times[0]],
                aspect='auto',
                cmap='viridis',
                interpolation='nearest',
            )
            if i == 0:
                ax.set_ylabel('Time (s)', fontsize=10)
                first_im = im

        # Plotting only the position GCC-PHAT maximum value
        else:
            max_indices = np.argmax(grid, axis=1)
            n_cols = grid.shape[1]
            n_rows = grid.shape[0]
            
            delay_values = np.linspace(-MAX_SAMPLE_DELAY*(1e6 / fs), MAX_SAMPLE_DELAY*(1e6 / fs), n_cols)
            time_values  = np.linspace(times[0], times[-1], n_rows)
            max_delays   = delay_values[max_indices]

            ax.scatter(max_delays, time_values, s=5, c='steelblue', zorder=3)
            ax.set_xlim(-MAX_SAMPLE_DELAY*(1e6 / fs), MAX_SAMPLE_DELAY*(1e6 / fs))
            ax.set_ylim(times[-1], times[0])
            ax.set_ylabel('Time (s)', fontsize=10)

            
        ax.set_title(lbl)
        ax.set_xlabel('Delay ($\mu$s)', fontsize=10)
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
        fig.colorbar(first_im, cax=cbar_ax, label='GCC_PATH peak value')
        cbar_ax.yaxis.set_ticks_position('left')
        cbar_ax.yaxis.set_label_position('left')


    # ==================== Plot Bearing angle ====================
    ax = axes[3]
    ax.scatter(estimated_bearing,times,color='red',s=0.5) 
    ax.set_xlim(0, 360)
    ax.set_title("$\\theta$")
    ax.set_xlabel(f'Degrees (°)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.savefig("delay_bearing_estimation.png")
    plt.show()