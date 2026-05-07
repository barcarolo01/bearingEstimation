import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch
import numpy as np

filename = 'sig1afterFilt.wav'

_, originalData = wavfile.read('AudioFiles/0958_crop.wav')
originalData = originalData[:, 0]
samplingFrequency, signalData = wavfile.read(filename)

#signalData = signalData[:,2]
    
widthHeight = (12, 8)
plt.figure(figsize=widthHeight)
Pxx, freqs, bins, im = plt.specgram(signalData,Fs=samplingFrequency,NFFT=512,scale='dB')
plt.xlabel('Time')
plt.title(filename)
plt.ylabel('Frequency')
plt.savefig(f'{filename}.png')

# Verifica attenuazione nella passband
f, Porig = welch(originalData, samplingFrequency, nperseg=8192)
f, Pfilt = welch(signalData, samplingFrequency, nperseg=8192)
mask = (f > 1000) & (f < 15000)
print(f"RMS originale:  {np.sqrt(np.mean(Porig[mask])):.6f}")
print(f"RMS filtrato:   {np.sqrt(np.mean(Pfilt[mask])):.6f}")

plt.figure(figsize=widthHeight)
plt.plot(f, 10 * np.log10(Porig))
#plt.ylabel('PSD (dB/Hz)')

plt.show()