import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch
import numpy as np

filename = 'AudioFiles/0958_crop.wav'

samplingFrequency, signalData = wavfile.read(filename)
signalData = signalData[:, 0]

# Keep only one track
#signalData = signalData[:,2]
    
widthHeight = (12, 8)
plt.figure(figsize=widthHeight)
Pxx, freqs, bins, im = plt.specgram(signalData,Fs=samplingFrequency,NFFT=512,scale='dB')
plt.xlabel('Time')
plt.title(filename)
plt.ylabel('Frequency')
#plt.savefig(f'{filename}.png')

plt.show()