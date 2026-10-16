import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from digitalshadow.utils.gcc_phat import *
from digitalshadow.utils.utils import compute_sample_delay_array

# Input wav
fs, sig1 = wav.read('simulation0/SynthTracks/T0_F1_H1.wav')
_, sig2 = wav.read('simulation0/SynthTracks/T0_F1_H2.wav')
_, sig3 = wav.read('simulation0/SynthTracks/T0_F1_H3.wav')

samples_per_window = int(0.05 * fs)
d = 0.3
quality_threshold = 0.0
overlap = 0.5

delay_arr21, sample_delay_21, peak_21, times  = compute_sample_delay_array(sig2,sig1,fs,samples_per_window,d,quality_threshold=quality_threshold,overlap=overlap)
delay_arr32, sample_delay_32, peak_32, _ = compute_sample_delay_array(sig3,sig2,fs,samples_per_window,d,quality_threshold=quality_threshold,overlap=overlap)
delay_arr31, sample_delay_31, peak_31, _ = compute_sample_delay_array(sig3,sig1,fs,samples_per_window,d,quality_threshold=quality_threshold,overlap=overlap)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
fig.subplots_adjust(left=0.05, right=0.95, wspace=0.1)

data = [delay_arr21,delay_arr32,delay_arr31]
datasets_label = ["$\\tau_{21}$","$\\tau_{32}$","$\\tau_{31}$"]

fontsize = 15
for i, (ax, values, lbl) in enumerate(zip(axes, data, datasets_label)):
    mean_peaks = np.mean(values,axis=0)
    idx_of_center = int(len(mean_peaks)//2)
    x_axis = np.arange(-idx_of_center, idx_of_center+1)  
    mean_peaks_shifted = np.roll(mean_peaks, -idx_of_center)
    ax.plot(x_axis,mean_peaks,c='#55C56d', linewidth=4)
    ax.set_xlabel("Delay (samples)",fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.grid(ls="--",alpha=0.5)
    ax.set_title(lbl,fontsize = fontsize+4)
    ax.set_xlim([-25,25])

    # View the y label only on the left of the first subplot
    if i == 0:
        ax.set_ylabel("GCC-PHAT",fontsize=fontsize)
    
plt.show()
#fig.savefig('gcc_phat_plot.png', format='png')