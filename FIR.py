import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from utils_filters import *
import numpy as np
from utils_filters import *
from utils import *
from scipy.signal import firwin, lfilter
import numpy as np

filename = 'Audiofiles/0958_.wav'

def FIR_zero_phase(signal,fs,f_low,f_high,N=201): 
    # Finestra di Kaiser: buon compromesso attenuazione/larghezza lobo
    coeffs = firwin(N, [f_low, f_high],
                    window='hamming',
                    pass_zero=False,   # bandpass
                    fs=96000)
    # --- Offline (elaborazione a blocco) ---
    signal_filtered = filtfilt(coeffs, 1.0, signal)

    # Compensare il ritardo di (N-1)//2 campioni:
    #delay = (N - 1) // 2
    #signal_processed = signal_filtered[delay:]          # segnale allineato temporalmente
    return signal_filtered
    

if __name__ == "__main__":
    filename = 'Audiofiles/0958_.wav'
    fs, signal = wav.read(filename)
    f_low = 10
    f_high = 40000
    N = 501

    
    # 3-channels audio
    sig1_processed = FIR_zero_phase(signal[:,0],fs,f_low,f_high,N)
    sig2_processed = FIR_zero_phase(signal[:,1],fs,f_low,f_high,N)
    sig3_processed = FIR_zero_phase(signal[:,2],fs,f_low,f_high,N)
    processed_joined = np.column_stack((sig1_processed, sig2_processed,sig3_processed))
    wav.write(f"{filename[:-4]}clean.wav",fs,processed_joined.astype(np.int16))

    '''

    # Mono
    processed = FIR_zero_phase(signal,fs,f_low,f_high,N)
    wav.write(f"{filename[:-4]}clean.wav",fs,processed.astype(np.int16))
    '''