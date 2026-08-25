from matplotlib.pylab import rfft, rfftfreq
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import butter, find_peaks, get_window, lfilter, decimate, hilbert, welch
from math import floor
from scipy.signal import detrend, butter, sosfiltfilt


def decimate_cascade(x, factor, ftype='fir', max_step=10):
    """
    Applica scipy.signal.decimate a cascata quando 'factor' e' troppo
    grande per una singola chiamata (scipy sconsiglia factor > 13).
    Cerca ad ogni passo il divisore esatto di 'factor' <= max_step, cosi'
    il prodotto degli step ricostruisce esattamente il fattore richiesto.
    """
    x = np.asarray(x)
    remaining = factor
    while remaining > 1:
        step = min(remaining, max_step)
        while remaining % step != 0:
            step -= 1
        x = decimate(x, step, ftype=ftype)
        remaining //= step
    return x


def square_law(x, cutoff=1000.0, high=30000, low=20000, fs=200000):
    """
    DEMON - metodo "square law".
    :param x: numpy.ndarray, segnale grezzo
    :param cutoff: frequenza max di interesse nello spettro DEMON (Hz)
    :param high: limite superiore banda passante (Hz)
    :param low: limite inferiore banda passante (Hz)
    :param fs: sample rate del segnale originale (Hz)
    :return: (envelope, new_fs)
    """
    if (high + low) / 2 <= 2 * (high - low):
        raise Exception("Errore: la larghezza di banda eccede la frequenza centrale della banda passante")

    nyq = 0.5 * fs
    high_n = high / nyq
    low_n = low / nyq
    order = 3

    b, a = butter(order, [low_n, high_n], btype='band')
    # Filtro bandpass CAUSALE (lfilter), come nell'originale.
    # Nota: introduce uno sfasamento dipendente dalla frequenza; se serve
    # fase zero si puo' sostituire con sosfiltfilt, ma questo si discosta
    # dall'algoritmo pubblicato.
    x = lfilter(b, a, x)

    x = np.square(x)

    n = int(floor(fs / (cutoff * 2)))
    if n < 1:
        raise ValueError(f"cutoff troppo alto rispetto a fs: n={n}")
    x = decimate_cascade(x, n, ftype='fir')

    x = np.sqrt(x)
    x = x - np.mean(x)

    new_fs = fs / n
    return x, new_fs


def hilbert_detector(x, target_fs=None, cutoff=1000.0, high=30000, low=20000, fs=200000):
    ...
    if target_fs is not None:
        n = int(floor(fs / target_fs))
        print(f"n is {n} [NOT None]")
    else:
        n = int(floor(fs / (cutoff * 2)))
        print(f"n is {n}")
    """
    DEMON - metodo "hilbert transform".
    :param x: numpy.ndarray, segnale grezzo
    :param cutoff: frequenza max di interesse nello spettro DEMON (Hz)
    :param high: limite superiore banda passante (Hz)
    :param low: limite inferiore banda passante (Hz)
    :param fs: sample rate del segnale originale (Hz)
    :return: (envelope, new_fs)
    """
    nyq = 0.5 * fs
    high_n = high / nyq
    low_n = low / nyq
    order = 3

    b, a = butter(order, [low_n, high_n], btype='band')
    x = lfilter(b, a, x)  # bandpass causale, come nell'originale

    x = hilbert(x)
    x = np.abs(x)

    n = int(floor(fs / (cutoff * 2)))
    if n < 1:
        raise ValueError(f"cutoff troppo alto rispetto a fs: n={n}")
    x = decimate_cascade(x, n, ftype='fir')

    # sqrt() e' presente nell'originale anche per l'hilbert detector,
    # nonostante l'inviluppo (abs di hilbert) non sia una quantita' al
    # quadrato come nel caso square_law. Mantenuto per fedelta' al
    # repository ufficiale.
    x = np.sqrt(x)
    x = x - np.mean(x)

    new_fs = fs / n
    return x, new_fs
 
def find_demon_peaks(freqs, psd, freq_min=0.5, freq_max=None,
                      prominence_db=6.0, max_peaks=10,
                      min_psd=None):
    """
    :param min_psd: soglia assoluta minima di PSD (scala lineare) sotto la
                     quale un picco viene scartato, indipendentemente dalla
                     sua prominenza. None = nessuna soglia assoluta.
    """
    mask = freqs >= freq_min
    if freq_max is not None:
        mask &= freqs <= freq_max

    f = freqs[mask]
    p = psd[mask]
    p_db = 10 * np.log10(p + 1e-20)

    height_db = None
    if min_psd is not None:
        height_db = 10 * np.log10(min_psd + 1e-20)  # conversione corretta (potenza)

    idx, props = find_peaks(p_db, prominence=prominence_db, height=height_db)
    if len(idx) == 0:
        return np.array([]), np.array([])

    order = np.argsort(props['prominences'])[::-1][:max_peaks]
    idx_sorted = idx[order]

    peak_freqs = f[idx_sorted]
    peak_psd = p[idx_sorted]

    order_f = np.argsort(peak_freqs)
    return peak_freqs[order_f], peak_psd[order_f]


def compute_demon_spectrum(envelope, env_fs):
    """
    Calcola lo spettro DEMON dell'inviluppo con Welch.
    """
    nperseg = min(len(envelope), int(env_fs / 0.1))
    nperseg = max(nperseg, 128)
    noverlap = nperseg // 2
    nfft = nperseg * 2

    freqs, psd = welch(envelope, fs=env_fs, nperseg=nperseg,
                        noverlap=noverlap, nfft=nfft,
                        window='hann', scaling='density')
    return freqs, psd



if __name__ == "__main__":

    # --------- Parametri da adattare ---------
    WAV_PATH = "AudioFiles/barca.wav"
    #WAV_PATH = "AudioFiles/0937_crop.wav"
    LOW_HZ = 10.0         # limite inferiore banda passante (Hz)
    HIGH_HZ = 25000.0       # limite superiore banda passante (Hz)
    CUTOFF_HZ = 500.0       # frequenza massima di interesse nello spettro DEMON (Hz)
    FREQ_PLOT_MAX = 300.0    # range in Hz mostrato nel plot dello spettro
    # ------------------------------------------

    fs, sig0 = wav.read(WAV_PATH)
    
    sig0 = sig0.astype(np.float64)

    # Verifica coerenza banda passante / Nyquist
    nyq = fs / 2
    if HIGH_HZ >= nyq:
        raise ValueError(f"high={HIGH_HZ} Hz >= Nyquist={nyq} Hz: abbassa 'high' o aumenta fs")

    envelope, env_fs = hilbert_detector(sig0, 10000, cutoff=CUTOFF_HZ,high=HIGH_HZ, low=LOW_HZ, fs=fs)


    # oppure, più aggressivo: high-pass l'inviluppo prima della Welch,
    # per tagliare via qualunque deriva sub-Hz
    sos_hp = butter(3, 1.5, btype='high', fs=env_fs, output='sos')
    envelope_hp = sosfiltfilt(sos_hp, envelope)

    freqs, psd = compute_demon_spectrum(envelope_hp, env_fs)
    mask = freqs <= FREQ_PLOT_MAX

     # --------- Rilevamento picchi ---------
    PEAK_FREQ_MIN = 0.5       # esclude drift residuo vicino a 0
    PEAK_PROMINENCE_DB = 6  # abbassa se mancano picchi veri, alza se ne prende troppi/rumore
    PEAK_MAX_COUNT = 10
    FREQ_PLOT_MAX = 300.0     # alzato per mostrare bene la serie 70-140-210-280

    peak_freqs, peak_psd = find_demon_peaks(
    freqs, psd,
    freq_min=PEAK_FREQ_MIN,
    freq_max=FREQ_PLOT_MAX,
    prominence_db=PEAK_PROMINENCE_DB,  # es. 3-6 dB, filtra "quanto emerge"
    min_psd=0.08,                        # filtra "quanto vale in assoluto"
    max_peaks=PEAK_MAX_COUNT,
)
 
    if len(peak_freqs) > 0:
        print("Picchi rilevati (Hz):")
        for pf, pp in zip(peak_freqs, peak_psd):
            print(f"  {pf:7.2f} Hz   PSD={pp:.4g}")
    else:
        print("Nessun picco significativo rilevato con i parametri correnti.")

    # ================== PLOTTING ==================
    FONTSIE = 15
    plt.rcParams.update({
        "font.size": FONTSIE,
    })
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    ax = axes[0]
    t = np.arange(len(envelope)) / env_fs
    ax.plot(t, envelope, color='red', linewidth=0.4)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Decimated envelope")



    ax = axes[1]
    ax.plot(freqs[mask], psd[mask], color='green', linewidth=0.8,zorder = 1)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("DEMON Spectrum")

    peak_freqs = peak_freqs[1:]
    peak_psd = peak_psd[1:]

    for f, p in zip(peak_freqs,peak_psd):
        ax.scatter(f, p,marker='o',color='red')
        ax.annotate(f"{f} Hz", (f, p), xytext=(7,-10), textcoords='offset points', color='red',zorder=5)

    ax.scatter(34, 0.07451,marker='o',color='orange')
    ax.annotate(f"{34.0} Hz", (34, 0.07451), xytext=(7,-10), textcoords='offset points', color='orange',zorder=5)


    plt.tight_layout()
    #plt.savefig("DEMON_Garda.png")
    plt.show()