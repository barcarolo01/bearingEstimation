# Source - https://stackoverflow.com/a/55312765
# Posted by Sheldon, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-16, License - CC BY-SA 4.0

import matplotlib.pyplot as plt
from scipy.io import wavfile

samplingFrequency, signalData = wavfile.read('AudioFiles/0958_crop.wav')
signalData = signalData[:,0]

plt.title('Spectrogram')    
Pxx, freqs, bins, im = plt.specgram(signalData,Fs=samplingFrequency,NFFT=512)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()
#plt.savefig('Spettrogramma.png')