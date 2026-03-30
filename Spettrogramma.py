import matplotlib.pyplot as plt
from scipy.io import wavfile

samplingFrequency, signalData = wavfile.read('AudioFiles/0958_crop.wav')
signalData = signalData[:,0]

plt.title('Spectrogram')    
Pxx, freqs, bins, im = plt.specgram(signalData,Fs=samplingFrequency,NFFT=512)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()