import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.io import wavfile
from scipy.signal import welch
import numpy as np
font_size = 13
filename = '../AudioFiles/0958_crop.wav'

samplingFrequency, signalData = wavfile.read(filename)
signalData = signalData[:, 0]
signalData = signalData / np.iinfo(signalData.dtype).max


widthHeight = (16, 5)
fig, ax = plt.subplots(figsize=widthHeight)

Pxx, freqs, bins, im = ax.specgram(signalData, Fs=samplingFrequency, NFFT=4096, scale='dB', cmap='magma')

cbar = fig.colorbar(im, ax=ax, location='right', pad=0.05)
cbar.set_label('dB', labelpad=5, fontsize=font_size)
im.set_clim(-150,-60)
cbar.ax.tick_params(axis='both', which='major', labelsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=font_size)
ax.set_xlabel('Time (s)',fontsize=font_size)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/1000)}'))
ax.set_ylabel('Frequency (kHz)',labelpad=15,fontsize=font_size)
plt.savefig(f'spectrogram_real.png', bbox_inches='tight')
plt.show()