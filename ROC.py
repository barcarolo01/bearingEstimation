import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
from scipy.ndimage import median_filter
from scipy.signal import butter, decimate, hilbert, sosfilt, sosfiltfilt, welch

# ======================= PARAMETRI DA ADATTARE =======================
TRACKS_GLOB = "AudioFiles/*.wav"  # tracce sintetiche SENZA rumore
CHANNEL = 0             # canale usato se le tracce sono multicanale
SEG_S = 1.0             # durata di una prova (s)

# Elaborazione DEMON (stessi valori del detector usato nel simulatore)
LOW_HZ = 2.0            # limite inferiore banda passante (Hz)
HIGH_HZ = 15000.0       # limite superiore banda passante (Hz)
CUTOFF_HZ = 500.0       # frequenza max dello spettro DEMON (Hz)
APPLY_SQRT = True       # sqrt finale come nel repository pyDEMON

# Spettro e normalizzazione
WELCH_SEG_S = 0.25      # segmento Welch (s): 0.25 s -> ~7 medie su 1 s
BG_WIN_HZ = 30.0        # finestra del filtro mediano per il fondo (Hz)

# Statistiche di test
FMIN, FMAX = 5.0, 300.0     # banda di ricerca nello spettro DEMON (Hz)
F0_RANGE = (10.0, 150.0)    # fondamentali candidate per la somma armonica (Hz)
N_HARM = 3                  # numero di armoniche sommate

# Monte Carlo
SNR_DB_LIST = [-15, -10, -5, 0, 5]  # adatta: serve l'intervallo in cui Pd passa da ~0 a ~1
N_H0 = 1000             # prove solo rumore (>= 10 / PFA_TARGET)
N_H1 = 300              # prove con segnale, per ogni SNR
PFA_TARGET = 0.01       # Pfa per la curva Pd vs SNR
SEED = 0

# Rumore: True = rumore colorato di prova (solo per testare lo script).
# Per i risultati della tesi collega ambient_noise in generate_noise() e metti False.
USE_TEST_NOISE = True

OUT_PREFIX = "roc_demon"
# =====================================================================


# ------------------------- rumore ambientale -------------------------
def generate_noise(n_samples, fs, rng):
    """
    Deve restituire SOLO rumore (nessun segnale), lungo n_samples, a fs,
    con una realizzazione indipendente a ogni chiamata.
    """
    if USE_TEST_NOISE:
        return _test_colored_noise(n_samples, fs, rng)
    # Esempio di aggancio (adatta alla firma reale della tua funzione):
    # from ambient_noise import ambient_noise
    # return ambient_noise(n_samples=n_samples, fs=fs, seed=int(rng.integers(2**31)))
    raise NotImplementedError("Collega qui ambient_noise e imposta USE_TEST_NOISE = False")


def _test_colored_noise(n_samples, fs, rng, slope=-1.7, f_ref=10.0):
    """Rumore gaussiano con PSD ~ f^slope (andamento tipo Wenz). Solo per test."""
    spec = np.fft.rfft(rng.standard_normal(n_samples))
    f = np.fft.rfftfreq(n_samples, 1.0 / fs)
    shape = np.maximum(f, f_ref) ** (slope / 2.0)  # ampiezza = sqrt(PSD)
    return np.fft.irfft(spec * shape, n=n_samples)


# ----------------------------- tracce --------------------------------
def load_tracks(pattern, channel):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Nessun file trovato con il pattern: {pattern}")
    tracks, fs_ref = [], None
    for path in files:
        fs, x = wav.read(path)
        if x.ndim > 1:
            x = x[:, channel]
        if fs_ref is None:
            fs_ref = fs
        elif fs != fs_ref:
            raise ValueError(f"{path}: fs={fs} diversa da {fs_ref}")
        tracks.append(x.astype(np.float64))
    return tracks, fs_ref, files


def get_segment(x, n, rng):
    """Segmento di n campioni: tutta la traccia se e' lunga n, altrimenti un tratto casuale."""
    if len(x) < n:
        raise ValueError(f"Traccia piu' corta del segmento richiesto ({len(x)} < {n})")
    start = 0 if len(x) == n else int(rng.integers(0, len(x) - n + 1))
    return x[start:start + n]


# ------------------------------ DEMON --------------------------------
def decimate_cascade(x, factor, ftype="fir", max_step=10):
    """Decimazione a cascata (scipy sconsiglia fattori > 13 in una sola chiamata)."""
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


def hilbert_envelope(x, fs):
    """
    hilbert_detector di pyDEMON. Differenze: filtro in forma SOS (stessa
    risposta del Butterworth ordine 3 causale, ma numericamente stabile con
    LOW_HZ molto basso) e sqrt protetta dai piccoli valori negativi dovuti
    al ripple del filtro di decimazione.
    """
    sos = butter(3, [LOW_HZ, HIGH_HZ], btype="band", fs=fs, output="sos")
    x = np.abs(hilbert(sosfilt(sos, x)))
    n = int(np.floor(fs / (2.0 * CUTOFF_HZ)))
    if n < 1:
        raise ValueError("CUTOFF_HZ troppo alto rispetto a fs")
    x = decimate_cascade(x, n)
    if APPLY_SQRT:
        x = np.sqrt(np.maximum(x, 0.0))
    return x - np.mean(x), fs / n


def demon_normalized(envelope, env_fs):
    """Spettro DEMON (Welch) diviso per il fondo stimato con un filtro mediano."""
    nperseg = int(WELCH_SEG_S * env_fs)
    freqs, psd = welch(envelope, fs=env_fs, nperseg=nperseg, noverlap=nperseg // 2,
                       nfft=2 * nperseg, window="hann", scaling="density")
    df = freqs[1] - freqs[0]
    k = max(3, int(BG_WIN_HZ / df) | 1)  # finestra dispari, in bin
    background = median_filter(psd, size=k, mode="nearest")
    return freqs, psd / (background + 1e-30)


def detection_stats(freqs, r):
    """Restituisce {nome: (T, frequenza associata)} per le due statistiche."""
    df = freqs[1] - freqs[0]
    band = (freqs >= FMIN) & (freqs <= FMAX)
    i = np.argmax(r[band])
    out = {"peak": (r[band][i], freqs[band][i])}

    best_T, best_f0 = -np.inf, np.nan
    for f0 in np.arange(F0_RANGE[0], F0_RANGE[1] + df / 2, df):
        harm = [h * f0 for h in range(1, N_HARM + 1) if h * f0 <= FMAX]
        idx = np.round(np.array(harm) / df).astype(int)
        T = r[idx].mean()
        if T > best_T:
            best_T, best_f0 = T, f0
    out["harmonic"] = (best_T, best_f0)
    return out


def trial_statistics(x, fs):
    env, env_fs = hilbert_envelope(x, fs)
    f, r = demon_normalized(env, env_fs)
    return detection_stats(f, r)


# ------------------------------- SNR ---------------------------------
def band_power(x, fs):
    sos = butter(4, [max(LOW_HZ, 1.0), HIGH_HZ], btype="band", fs=fs, output="sos")
    return np.mean(sosfiltfilt(sos, x) ** 2)


def noise_gain(signal, noise, snr_db, fs):
    """Fattore g tale che SNR(signal, g*noise) nella banda passante = snr_db."""
    return np.sqrt(band_power(signal, fs) / (band_power(noise, fs) * 10 ** (snr_db / 10)))


# ------------------------------- ROC ---------------------------------
def roc_curve(T0, T1):
    """Pfa e Pd per tutte le soglie (decisione: T > soglia)."""
    thr = np.concatenate([[np.inf], np.unique(np.concatenate([T0, T1]))[::-1]])
    T0s, T1s = np.sort(T0), np.sort(T1)
    pfa = 1.0 - np.searchsorted(T0s, thr, side="right") / len(T0)
    pd = 1.0 - np.searchsorted(T1s, thr, side="right") / len(T1)
    pfa, pd = np.append(pfa, 1.0), np.append(pd, 1.0)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return pfa, pd, float(trapz(pd, pfa))


def pd_at_pfa(T0, T1, pfa):
    thr = np.quantile(T0, 1.0 - pfa)
    return float(np.mean(T1 > thr)), float(thr)


# ------------------------------- main --------------------------------
def main():
    rng = np.random.default_rng(SEED)
    tracks, fs, files = load_tracks(TRACKS_GLOB, CHANNEL)
    n = int(round(SEG_S * fs))
    if HIGH_HZ >= fs / 2:
        raise ValueError(f"HIGH_HZ={HIGH_HZ} >= Nyquist={fs / 2}")
    if USE_TEST_NOISE:
        print("ATTENZIONE: rumore di prova, non ambient_noise. Solo per verificare lo script.")
    print(f"{len(files)} tracce, fs = {fs} Hz, prove da {n} campioni")

    names = ("peak", "harmonic")

    # H0: solo rumore (indipendente dall'SNR grazie alla normalizzazione)
    T0 = {k: np.empty(N_H0) for k in names}
    for i in range(N_H0):
        st = trial_statistics(generate_noise(n, fs, rng), fs)
        for k in names:
            T0[k][i] = st[k][0]
        if (i + 1) % 200 == 0:
            print(f"  H0: {i + 1}/{N_H0}")

    # H1: segnale + rumore, per ogni SNR
    T1 = {k: np.empty((len(SNR_DB_LIST), N_H1)) for k in names}
    F1 = {k: np.empty((len(SNR_DB_LIST), N_H1)) for k in names}
    for j, snr in enumerate(SNR_DB_LIST):
        for i in range(N_H1):
            s = get_segment(tracks[i % len(tracks)], n, rng)
            noise = generate_noise(n, fs, rng)
            st = trial_statistics(s + noise_gain(s, noise, snr, fs) * noise, fs)
            for k in names:
                T1[k][j, i], F1[k][j, i] = st[k]
        print(f"  H1: SNR = {snr:+d} dB completato")

    # Risultati
    print(f"\n{'SNR':>6} | {'AUC picco':>9} {'Pd picco':>9} | {'AUC arm.':>9} {'Pd arm.':>9} | f0 mediana")
    pd_curves = {k: [] for k in names}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for j, snr in enumerate(SNR_DB_LIST):
        row = []
        for ax, k in zip(axes, names):
            pfa, pd, auc = roc_curve(T0[k], T1[k][j])
            pdv, _ = pd_at_pfa(T0[k], T1[k][j], PFA_TARGET)
            pd_curves[k].append(pdv)
            ax.step(pfa, pd, where="post", label=f"SNR = {snr} dB (AUC {auc:.2f})")
            row += [auc, pdv]
        print(f"{snr:>4} dB | {row[0]:9.3f} {row[1]:9.3f} | {row[2]:9.3f} {row[3]:9.3f} |"
              f" {np.median(F1['harmonic'][j]):6.1f} Hz")

    titles = {"peak": "Picco singolo", "harmonic": f"Somma armonica ({N_HARM} armoniche)"}
    for ax, k in zip(axes, names):
        diag = np.logspace(np.log10(1.0 / N_H0), 0, 200)  # Pd = Pfa (asse x logaritmico)
        ax.plot(diag, diag, "k--", lw=0.8, label="Caso (Pd = Pfa)")
        ax.set_xscale("log")
        ax.set_xlim(1.0 / N_H0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Pfa")
        ax.set_title(titles[k])
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(fontsize=9, loc="lower right")
    axes[0].set_ylabel("Pd")
    fig.suptitle("ROC del detector DEMON")
    fig.tight_layout()
    fig.savefig(f"{OUT_PREFIX}_roc.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for k in names:
        ax2.plot(SNR_DB_LIST, pd_curves[k], "o-", label=titles[k])
    ax2.set_xlabel("SNR nella banda passante (dB)")
    ax2.set_ylabel("Pd")
    ax2.set_ylim(0, 1.02)
    ax2.set_title(f"Pd vs SNR a Pfa = {PFA_TARGET}")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(f"{OUT_PREFIX}_pd_vs_snr.png", dpi=150)

    np.savez(f"{OUT_PREFIX}_results.npz", snr_db=np.array(SNR_DB_LIST),
             T0_peak=T0["peak"], T0_harmonic=T0["harmonic"],
             T1_peak=T1["peak"], T1_harmonic=T1["harmonic"],
             f_peak=F1["peak"], f0_harmonic=F1["harmonic"],
             files=np.array(files), pfa_target=PFA_TARGET)
    print(f"\nSalvati: {OUT_PREFIX}_roc.png, {OUT_PREFIX}_pd_vs_snr.png, {OUT_PREFIX}_results.npz")
    plt.show()


if __name__ == "__main__":
    main()