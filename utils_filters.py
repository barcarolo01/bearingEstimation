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



def add_gaussian_noise(signal, desired_SNR_dB):
    """
    Aggiunge rumore bianco gaussiano a un segnale per ottenere un SNR desiderato.
    
    Parameters:
        signal (np.ndarray): Il segnale di input (letto da wav.read).
        desired_SNR_dB (float): SNR desiderato in dB.
    
    Returns:
        np.ndarray: Segnale con rumore aggiunto, nello stesso dtype dell'originale.
    """
    # Conversione in float64
    sig_float = signal.astype(np.float64)

    signal_power = np.mean(sig_float ** 2)

    # Lineare --> dB
    SNR_linear = 10 ** (desired_SNR_dB / 10)

    # Potenza del rumore necessaria: P_noise = P_signal / SNR_linear
    noise_power = signal_power / SNR_linear
    noise = np.random.normal(0, np.sqrt(noise_power), sig_float.shape)
    noisy_signal = sig_float + noise

    # Convresione in int
    if np.issubdtype(signal.dtype, np.integer):
        info = np.iinfo(signal.dtype)
        noisy_signal = np.clip(noisy_signal, info.min, info.max)

    return noisy_signal.astype(signal.dtype)
