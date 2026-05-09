from gcc_phat import *
from scipy.signal import filtfilt, firwin

def FIR_bandpass_filter(signal, lowcut, highcut, fs, N=201):
    coeffs = firwin(
        N,
        [lowcut, highcut],
        window=('kaiser', 8.0),
        pass_zero=False,
        fs=fs
    )
    
    processed = filtfilt(coeffs,1.0, signal)
    return processed

def FIR_lowpass_filter(signal, fc, fs, N=201):
    coeffs = firwin(
        N,
        fc,
        #window=('kaiser', 8.0),
        window = 'hamming',
        pass_zero=False,
        fs=fs
    )
    
    processed = filtfilt(coeffs,1.0, signal)
    return processed


def lowpass_filter_fft(signal: np.ndarray, fs: float, fH: float):
    n = len(signal)
    spectrum = np.fft.fft(signal)

    frequencies = np.fft.fftfreq(n, d=1.0 / fs)
    
    spectrum[np.abs(frequencies) > fH] = 0.0
    #spectrum[frequencies > fH] = 0.0

    filtered_signal = np.fft.ifft(spectrum).real

    return filtered_signal
