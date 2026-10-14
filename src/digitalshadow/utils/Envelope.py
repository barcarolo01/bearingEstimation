import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.ndimage import median_filter
from scipy.signal import butter, decimate, find_peaks, hilbert, lfilter, sosfilt, sosfiltfilt, welch


# ======================= PARAMETRI DEL DETECTOR =======================
LOW_HZ = 5.0            # limite inferiore banda passante (Hz)
HIGH_HZ = 15000.0       # limite superiore banda passante (Hz)
CUTOFF_HZ = 500.0       # frequenza massima dello spettro DEMON (Hz)

F_MIN_HZ = 67.0         # finestra di ricerca del blade rate: [F_MIN_HZ, F_MAX_HZ] (Hz)
F_MAX_HZ = 73.0
HARMONICS = (1,)        # armoniche usate: (1,) solo la finestra base; (1, 2, 3) anche
                        # [2*F_MIN_HZ, 2*F_MAX_HZ] e [3*F_MIN_HZ, 3*F_MAX_HZ]
THRESHOLD = 9.0        # soglia sullo spettro normalizzato. Abbassarla aumenta la
                        # probabilita' di rivelazione e anche i falsi allarmi

SEG_S = None            # None = periodogramma su tutto il frame (come prima);
                        # es. 0.5 = Welch con segmenti da 0.5 s (meno varianza)
NORM = "mean"           # "mean"   = PSD / media globale (come prima)
                        # "median" = PSD / mediana locale scorrevole (piu' robusto)
BG_WIN_HZ = 30.0        # larghezza della mediana scorrevole se NORM = "median"

FREQ_PLOT_MIN = 10.0    # range mostrato nel plot (Hz)
FREQ_PLOT_MAX = 300.0
# =====================================================================


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
        if step == 1:            # fattore residuo primo > max_step: un solo passo
            step = remaining
        x = decimate(x, step, ftype=ftype)
        remaining //= step
    return x


def square_law(x, cutoff=1000.0, high=30000, low=20000, fs=200000):
    """
    DEMON - metodo "square law" (invariato).
    :return: (envelope, new_fs)
    """
    if (high + low) / 2 <= 2 * (high - low):
        raise Exception("Errore: la larghezza di banda eccede la frequenza centrale della banda passante")

    nyq = 0.5 * fs
    b, a = butter(3, [low / nyq, high / nyq], btype='band')
    x = lfilter(b, a, x)
    x = np.square(x)

    n = int(np.floor(fs / (cutoff * 2)))
    if n < 1:
        raise ValueError(f"cutoff troppo alto rispetto a fs: n={n}")
    x = decimate_cascade(x, n, ftype='fir')

    x = np.sqrt(np.maximum(x, 0.0))
    x = x - np.mean(x)
    return x, fs / n


def hilbert_detector(x, target_fs=None, cutoff=1000.0, high=30000, low=20000, fs=200000):
    """
    DEMON - metodo "hilbert transform".
    :param x: numpy.ndarray, segnale grezzo
    :param target_fs: se dato, fattore di decimazione = floor(fs / target_fs);
                      altrimenti floor(fs / (2 * cutoff)) come nell'originale
    :param cutoff: frequenza max di interesse nello spettro DEMON (Hz)
    :param high: limite superiore banda passante (Hz)
    :param low: limite inferiore banda passante (Hz)
    :param fs: sample rate del segnale originale (Hz)
    :return: (envelope, new_fs)

    Il Butterworth di ordine 3 e' applicato in forma SOS: stessa risposta
    del filtro causale originale (b, a), ma numericamente stabile anche con
    'low' di pochi Hz a fs = 96 kHz.
    """
    if target_fs is not None:
        n = int(np.floor(fs / target_fs))
    else:
        n = int(np.floor(fs / (cutoff * 2)))
    if n < 1:
        raise ValueError(f"fattore di decimazione non valido: n={n}")

    sos = butter(3, [low, high], btype='band', fs=fs, output='sos')
    x = sosfilt(sos, x)          # bandpass causale, come nell'originale

    x = np.abs(hilbert(x))
    x = decimate_cascade(x, n, ftype='fir')

    # sqrt() presente nell'originale anche per l'hilbert detector (vedi
    # discussione); max(.,0) evita NaN dovuti al ripple del filtro FIR.
    x = np.sqrt(np.maximum(x, 0.0))
    x = x - np.mean(x)
    return x, fs / n


def compute_demon_spectrum(envelope, env_fs, seg_s=None):
    """
    Spettro DEMON dell'inviluppo con Welch.
    seg_s=None -> un solo segmento lungo tutto il frame (periodogramma).
    """
    if seg_s is None:
        nperseg = min(len(envelope), int(env_fs / 0.1))
    else:
        nperseg = min(len(envelope), int(seg_s * env_fs))
    nperseg = max(nperseg, 128)
    freqs, psd = welch(envelope, fs=env_fs, nperseg=nperseg,
                       noverlap=nperseg // 2, nfft=nperseg * 2,
                       window='hann', scaling='density')
    return freqs, psd


def normalize_spectrum(freqs, psd, norm="mean", bg_win_hz=30.0):
    """
    Rende lo spettro indipendente dalla scala del segnale:
      "mean"   -> PSD / media globale
      "median" -> PSD / mediana scorrevole su bg_win_hz (stima del fondo locale)
    """
    if norm == "mean":
        return psd / np.mean(psd)
    if norm == "median":
        df = freqs[1] - freqs[0]
        k = max(3, int(bg_win_hz / df) | 1)          # finestra dispari, in bin
        return psd / (median_filter(psd, size=k, mode='nearest') + 1e-30)
    raise ValueError(f"norm sconosciuta: {norm}")


def detection_statistic(freqs, r, f_min=None, f_max=None, harmonics=None):
    """
    Per ogni armonica h: massimo dello spettro normalizzato in
    [h*f_min, h*f_max]. Statistica = media sulle armoniche.
    Restituisce (statistica, frequenza del massimo per ogni armonica).
    Se non specificati, usa F_MIN_HZ, F_MAX_HZ e HARMONICS.
    """
    f_min = F_MIN_HZ if f_min is None else f_min
    f_max = F_MAX_HZ if f_max is None else f_max
    harmonics = HARMONICS if harmonics is None else harmonics
    if not 0 < f_min < f_max:
        raise ValueError(f"finestra non valida: f_min={f_min}, f_max={f_max}")
    vals, peak_f = [], []
    for h in harmonics:
        m = (freqs >= h * f_min) & (freqs <= h * f_max)
        if not np.any(m):
            raise ValueError(f"finestra {h * f_min}-{h * f_max} Hz vuota o fuori dallo spettro")
        i = np.argmax(r[m])
        vals.append(r[m][i])
        peak_f.append(freqs[m][i])
    return float(np.mean(vals)), peak_f


def find_demon_peaks(freqs, psd, freq_min=0.5, freq_max=None, prominence_db=6.0, max_peaks=10):
    """Picchi per il plot (non usati per la decisione)."""
    mask = freqs >= freq_min
    if freq_max is not None:
        mask &= freqs <= freq_max
    f, p = freqs[mask], psd[mask]
    idx, props = find_peaks(10 * np.log10(p + 1e-20), prominence=prominence_db)
    if len(idx) == 0:
        return np.array([]), np.array([])
    idx = idx[np.argsort(props['prominences'])[::-1][:max_peaks]]
    order = np.argsort(f[idx])
    return f[idx][order], p[idx][order]


def detect_env(sig0, fs, threshold=THRESHOLD, plot=True, verbose=True, channel=0):
    """
    Detector DEMON. Restituisce True se nel frame e' presente il vessel,
    cioe' se lo spettro DEMON normalizzato supera 'threshold' nella finestra
    [F_MIN_HZ, F_MAX_HZ] (media sulle armoniche in HARMONICS).
    """
    sig0 = np.asarray(sig0, dtype=np.float64)
    if sig0.ndim > 1:
        sig0 = sig0[:, channel]

    if HIGH_HZ >= fs / 2:
        raise ValueError(f"high={HIGH_HZ} Hz >= Nyquist={fs / 2} Hz: abbassa 'high' o aumenta fs")

    # Inviluppo decimato
    envelope, env_fs = hilbert_detector(sig0, cutoff=CUTOFF_HZ, high=HIGH_HZ, low=LOW_HZ, fs=fs)

    # High-pass dell'inviluppo: toglie la deriva sub-Hz
    sos_hp = butter(3, 1.5, btype='high', fs=env_fs, output='sos')
    envelope_hp = sosfiltfilt(sos_hp, envelope)

    # Spettro DEMON normalizzato e decisione
    freqs, psd = compute_demon_spectrum(envelope_hp, env_fs, seg_s=SEG_S)
    r = normalize_spectrum(freqs, psd, norm=NORM, bg_win_hz=BG_WIN_HZ)
    stat, peak_f = detection_statistic(freqs, r)
    detected = stat > threshold

    if verbose:
        fs_txt = ", ".join(f"{f:.1f}" for f in peak_f)
        print(f"DEMON: statistica = {stat:.2f} (soglia {threshold}), "
              f"picco a {fs_txt} Hz -> {'VESSEL RIVELATO' if detected else 'nessun vessel'}")

    if plot:
        _plot(envelope, env_fs, freqs, r, stat, threshold, detected)

    return detected


def _plot(envelope, env_fs, freqs, r, stat, threshold, detected):
    plt.rcParams.update({"font.size": 15})
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    ax = axes[0]
    ax.plot(np.arange(len(envelope)) / env_fs, envelope, color='red', linewidth=0.4)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Decimated envelope")

    ax = axes[1]
    mask = (freqs >= FREQ_PLOT_MIN) & (freqs <= FREQ_PLOT_MAX)
    ax.plot(freqs[mask], r[mask], color='green', linewidth=0.8, zorder=1)
    for h in HARMONICS:
        ax.axvspan(h * F_MIN_HZ, h * F_MAX_HZ, color='orange', alpha=0.25)
    ax.axhline(threshold, color='k', linestyle='--', linewidth=0.8, label=f"soglia = {threshold}")

    pf, pp = find_demon_peaks(freqs, r, freq_min=FREQ_PLOT_MIN, freq_max=FREQ_PLOT_MAX,
                              prominence_db=6, max_peaks=3)
    for f, p in zip(pf, pp):
        ax.scatter(f, p, marker='o', color='red', zorder=5)
        ax.annotate(f"{f:.1f} Hz", (f, p), xytext=(7, -10), textcoords='offset points', color='red')

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized PSD")
    ax.set_title(f"DEMON Spectrum - statistica {stat:.1f} -> "
                 f"{'RIVELATO' if detected else 'non rivelato'}")
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("DEMON.png")
    plt.show()


if __name__ == "__main__":
    # --------- Parametri da adattare ---------
    WAV_PATH = "simulationDEMON/SynthTracks/T0_F1_H1.wav"

    # ------------------------------------------

    fs, sig0 = wav.read(WAV_PATH)
    detect_env(sig0, fs)