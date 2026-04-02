from gcc_phat import *
from scipy.signal import butter, filtfilt, firwin, lfilter, sosfilt

def lowpass_filter(signal, fc, fs, order=4):
    sos = butter(order, fc, btype='low', fs=fs, output='sos')
    return sosfilt(sos, signal)

def highpass_filter(signal, fc, fs, order=4):
    sos = butter(order, fc, btype='high', fs=fs, output='sos')
    return sosfilt(sos, signal)

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sosfilt(sos, signal)

def FIR_bandpass_filter(signal, lowcut, highcut, fs, N=201):
    """
    Filtro FIR bandpass a fase zero via filtfilt.
    Adatto per TDOA: nessun ritardo di gruppo introdotto.
    """
    coeffs = firwin(
        N,
        [lowcut, highcut],
        window=('kaiser', 8.0),
        pass_zero=False,
        fs=fs
    )

    # filtfilt: forward + backward → ritardo di gruppo = 0
    processed = filtfilt(coeffs, 1.0, signal)
    return processed



def add_gaussian_noise(signal, desired_SNR_dB):
    """
    Aggiunge rumore bianco gaussiano a un segnale per ottenere un SNR desiderato.
    
    Parameters:
        signal (np.ndarray): Il segnale di input (letto da wav.read).
        desired_SNR_dB (float): SNR desiderato in dB.
    
    Returns:
        np.ndarray: Segnale con rumore aggiunto, nello stesso dtype dell'originale.
    """
    # Converte in float64 per i calcoli
    sig_float = signal.astype(np.float64)

    # Potenza del segnale (media dei quadrati)
    signal_power = np.mean(sig_float ** 2)

    # SNR lineare da dB: SNR_linear = 10^(SNR_dB / 10)
    SNR_linear = 10 ** (desired_SNR_dB / 10)

    # Potenza del rumore necessaria: P_noise = P_signal / SNR_linear
    noise_power = signal_power / SNR_linear

    # Genera rumore gaussiano con la potenza calcolata
    # std = sqrt(P_noise) poiché P = E[x^2] = std^2 per media zero
    noise = np.random.normal(0, np.sqrt(noise_power), sig_float.shape)

    # Somma segnale + rumore
    noisy_signal = sig_float + noise

    # Riconduce al dtype originale (es. int16 per wav standard)
    if np.issubdtype(signal.dtype, np.integer):
        info = np.iinfo(signal.dtype)
        noisy_signal = np.clip(noisy_signal, info.min, info.max)

    return noisy_signal.astype(signal.dtype)