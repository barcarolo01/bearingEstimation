
"""
roc_detect.py - Curve ROC del detector DEMON definito in demon_detect.py

Usa ESATTAMENTE l'elaborazione di demon_detect.py (stessi parametri:
banda, finestra [F_MIN_HZ, F_MAX_HZ], armoniche, SEG_S, NORM, THRESHOLD),
quindi la ROC descrive il detector che usi nel simulatore.

Prove H0: solo rumore (realizzazione nuova a ogni prova).
Prove H0 e H1 usano generate_noise_array del modulo del rumore di Wenz
(lo stesso generatore di add_ambient_noise), con realizzazioni indipendenti.
Prove H1: traccia sintetica senza rumore + rumore scalato per ottenere l'SNR
          richiesto. Di default l'SNR e' definito come in add_ambient_noise
          (banda piena, potenza del segnale con _active_power).

La statistica e' normalizzata (invariante alla scala del segnale), quindi le
prove H0 sono calcolate una volta sola e valgono per tutti gli SNR.

Output:
  <OUT_PREFIX>_roc.png        ROC per ogni SNR, con il punto di lavoro (soglia THRESHOLD)
  <OUT_PREFIX>_pd_vs_snr.png  Pd in funzione dell'SNR alla soglia THRESHOLD
  <OUT_PREFIX>_results.npz    statistiche grezze, per rifare i grafici senza rilanciare
"""

import glob
from digitalshadow import config
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import butter, sosfiltfilt
from digitalshadow.underwater_sim.ambient_noise import *
import digitalshadow.underwater_sim.ambient_noise as AN
import digitalshadow.utils.Envelope as D

FONTSIZE = 16

# ======================= PARAMETRI DA ADATTARE =======================
TRACKS_GLOB = "simulationGarda/SynthTracks/*.wav"  # tracce sintetiche SENZA rumore
CHANNEL = 0             # canale usato se le tracce sono multicanale
SEG_S = 1.0             # durata di una prova (s), come i frame del simulatore

SNR_DB_LIST = [-15,-12,-9,-6,-3,0]  # SNR (dB), definito come in SNR_DEF
N_H0 = 1000             # prove solo rumore
N_H1 = 1000              # prove con segnale, per ogni SNR
SEED = 0

# Rumore di Wenz: generate_noise_array di digitalshadow.underwater_sim.ambient_noise
# (lo stesso generatore usato da add_ambient_noise). Di default i parametri vengono
# dalle stesse variabili d'ambiente del simulatore (NOISE_SHIPPING, NOISE_WIND, ...).


N_CH = 1                # canali generati: per il detector su un solo idrofono basta 1
C_SOUND = 1500.0
COHERENT = False        # come add_ambient_noise quando coherent=None

# Definizione dell'SNR:
#   "fullband" = come add_ambient_noise: banda piena [0, fs/2], potenza del segnale
#                misurata con _active_power (confrontabile con il -9 dB della GCC-PHAT)
#   "band"     = rapporto di potenze nella banda passante del detector [LOW_HZ, HIGH_HZ]
SNR_DEF = "fullband"

# True = rumore colorato di prova invece del tuo modulo (solo per verificare lo script)
USE_TEST_NOISE = False

OUT_PREFIX = "roc_detect"
ROC_XSCALE = "linear"   # "linear" = assi lineari (ROC classica 0-1); "log" = Pfa logaritmica
# =====================================================================


# ------------------------- rumore ambientale -------------------------
def noise_frame(n_samples, fs, rng):
    """
    SOLO rumore di Wenz (nessun segnale), lungo n_samples, sul canale CHANNEL.
    Stessa chiamata che fa add_ambient_noise; rng avanza a ogni chiamata, quindi
    ogni prova ha una realizzazione indipendente.
    (Si chiama noise_frame per non sovrascrivere generate_noise del tuo modulo.)
    """
    if USE_TEST_NOISE:
        return _test_colored_noise(n_samples, fs, rng)
    ch = CHANNEL if N_CH > 1 else 0
    noises = generate_noise_array(n_samples, fs, N_CH, config.NOISE_SPACING, None, C_SOUND,
                                  config.NOISE_SHIPPING, config.NOISE_WIND, rng, config.NOISE_EXTRA_DB, COHERENT)
    return np.asarray(noises[ch], dtype=np.float64)


def _test_colored_noise(n_samples, fs, rng, slope=-1.7, f_ref=10.0):
    """Rumore gaussiano con PSD ~ f^slope (andamento tipo Wenz). Solo per test."""
    spec = np.fft.rfft(rng.standard_normal(n_samples))
    f = np.fft.rfftfreq(n_samples, 1.0 / fs)
    return np.fft.irfft(spec * np.maximum(f, f_ref) ** (slope / 2.0), n=n_samples)


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
    """Tutta la traccia se lunga n campioni, altrimenti un tratto casuale di n campioni."""
    if len(x) < n:
        raise ValueError(f"Traccia piu' corta del frame richiesto ({len(x)} < {n})")
    start = 0 if len(x) == n else int(rng.integers(0, len(x) - n + 1))
    return x[start:start + n]


# ------------------- statistica (identica a detect_env) --------------
def statistic(x, fs):
    """Stesso calcolo di demon_detect.detect_env, ma restituisce la statistica."""
    env, env_fs = D.hilbert_detector(x, cutoff=D.CUTOFF_HZ, high=D.HIGH_HZ, low=D.LOW_HZ, fs=fs)
    sos_hp = butter(3, 1.5, btype='high', fs=env_fs, output='sos')
    env = sosfiltfilt(sos_hp, env)
    freqs, psd = D.compute_demon_spectrum(env, env_fs, seg_s=D.SEG_S)
    r = D.normalize_spectrum(freqs, psd, norm=D.NORM, bg_win_hz=D.BG_WIN_HZ)
    return D.detection_statistic(freqs, r)[0]


# ------------------------------- SNR ---------------------------------
def band_power(x, fs):
    sos = butter(4, [D.LOW_HZ, D.HIGH_HZ], btype='band', fs=fs, output='sos')
    return np.mean(sosfiltfilt(sos, x) ** 2)


def signal_noise_power(signal, noise, fs):
    """Potenze di segnale e rumore secondo SNR_DEF."""
    if SNR_DEF == "fullband" and not USE_TEST_NOISE:
        # stessa definizione di add_ambient_noise
        return AN._active_power(signal, fs), np.mean(noise ** 2)
    if SNR_DEF in ("fullband", "band"):
        if SNR_DEF == "fullband":   # rumore di prova: niente _active_power, banda piena semplice
            return np.mean(signal ** 2), np.mean(noise ** 2)
        return band_power(signal, fs), band_power(noise, fs)
    raise ValueError(f"SNR_DEF deve essere 'fullband' o 'band', non {SNR_DEF!r}")


def noise_gain(signal, noise, snr_db, fs):
    """g tale che l'SNR di signal + g*noise (definito come in SNR_DEF) sia snr_db."""
    ps, pn = signal_noise_power(signal, noise, fs)
    return np.sqrt(ps / (pn * 10 ** (snr_db / 10)))


# ------------------------------- ROC ---------------------------------
def roc_curve(T0, T1):
    """Pfa e Pd al variare della soglia (decisione: T > soglia), e AUC."""
    thr = np.concatenate([[np.inf], np.unique(np.concatenate([T0, T1]))[::-1]])
    pfa = 1.0 - np.searchsorted(np.sort(T0), thr, side='right') / len(T0)
    pd = 1.0 - np.searchsorted(np.sort(T1), thr, side='right') / len(T1)
    pfa, pd = np.append(pfa, 1.0), np.append(pd, 1.0)
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    return pfa, pd, float(trapz(pd, pfa))


def wilson_ci(k, n, z=1.96):
    """Intervallo di confidenza al 95% per una proporzione k/n."""
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


# ------------------------------- main --------------------------------
def main():
    rng = np.random.default_rng(SEED)
    tracks, fs, files = load_tracks(TRACKS_GLOB, CHANNEL)
    n = int(round(SEG_S * fs))
    thr = D.THRESHOLD
    if USE_TEST_NOISE:
        print("ATTENZIONE: rumore di prova, non il tuo rumore di Wenz. Solo per verificare lo script.")
    else:
        print(f"Rumore: generate_noise_array (shipping={config.NOISE_SHIPPING}, wind={config.NOISE_WIND}, "
              f"extra_db={config.NOISE_EXTRA_DB}); SNR {SNR_DEF}")
    print(f"{len(files)} tracce, fs = {fs} Hz, frame da {n} campioni")
    print(f"Detector: banda {D.LOW_HZ}-{D.HIGH_HZ} Hz, finestra {D.F_MIN_HZ}-{D.F_MAX_HZ} Hz, "
          f"armoniche {D.HARMONICS}, SEG_S={D.SEG_S}, NORM={D.NORM}, soglia={thr}")

    # H0: solo rumore
    T0 = np.empty(N_H0)
    for i in range(N_H0):
        T0[i] = statistic(noise_frame(n, fs, rng), fs)
        if (i + 1) % 200 == 0:
            print(f"  H0: {i + 1}/{N_H0}")
    k0 = int(np.sum(T0 > thr))
    pfa_thr = k0 / N_H0
    lo0, hi0 = wilson_ci(k0, N_H0)

    # H1: segnale + rumore, per ogni SNR
    T1 = np.empty((len(SNR_DB_LIST), N_H1))
    for j, snr in enumerate(SNR_DB_LIST):
        for i in range(N_H1):
            s = get_segment(tracks[i % len(tracks)], n, rng)
            noise = noise_frame(n, fs, rng)
            T1[j, i] = statistic(s + noise_gain(s, noise, snr, fs) * noise, fs)
        print(f"  H1: SNR = {snr:+d} dB completato")

    # Tabella risultati
    print(f"\nPfa alla soglia {thr}: {pfa_thr:.4f}  (IC 95%: {lo0:.4f}-{hi0:.4f})")
    print(f"{'SNR':>7} | {'AUC':>6} | {'Pd @ soglia':>11} | IC 95%")
    pd_thr, pd_lo, pd_hi = [], [], []
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SNR_DB_LIST)))
    for j, snr in enumerate(SNR_DB_LIST):
        pfa, pd, auc = roc_curve(T0, T1[j])
        k1 = int(np.sum(T1[j] > thr))
        lo, hi = wilson_ci(k1, N_H1)
        pd_thr.append(k1 / N_H1); pd_lo.append(lo); pd_hi.append(hi)
        print(f"{snr:>4} dB | {auc:6.3f} | {k1 / N_H1:11.3f} | {lo:.2f}-{hi:.2f}")

        ax.step(pfa, pd, where='post', color=colors[j], label=f"SNR = {snr} dB (AUC {auc:.2f})")
        x_op = max(pfa_thr, 1.0 / N_H0) if ROC_XSCALE == "log" else pfa_thr
        ax.plot(x_op, k1 / N_H1, 'o', color=colors[j], ms=7, mec='k')

    if ROC_XSCALE == "log":
        pfa_min = 1.0 / N_H0                  # su scala log Pfa = 0 non e' rappresentabile
        diag = np.logspace(np.log10(pfa_min), 0, 200)
        ax.set_xscale('log')
        ax.set_xlim(pfa_min, 1)
    elif ROC_XSCALE == "linear":
        pfa_min = 0.0
        diag = np.array([0.0, 1.0])
        ax.set_xlim(0, 1)
        ax.set_aspect('equal')
    else:
        raise ValueError(f"ROC_XSCALE deve essere 'linear' o 'log', non {ROC_XSCALE!r}")
    #ax.plot(diag, diag, 'k--', lw=0.8, label="Caso (Pd = Pfa)")
    ax.axvline(max(pfa_thr, pfa_min), color='gray', lw=0.8, ls=':',
               label=f"T = {thr} (Pfa = {pfa_thr:.3f})")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False alarm probability (Pfa)",fontsize=FONTSIZE)
    ax.set_ylabel("Detection probability (Pd)",fontsize=FONTSIZE)
    #ax.set_title("ROC del detector DEMON")
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(loc='lower right',fontsize=FONTSIZE-2)
    plt.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    plt.tick_params(axis="both", which="minor", labelsize=FONTSIZE)
    fig.tight_layout()
    fig.savefig(f"{OUT_PREFIX}_roc.pdf")

    # Pd vs SNR alla soglia scelta
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    pd_thr, pd_lo, pd_hi = map(np.array, (pd_thr, pd_lo, pd_hi))
    yerr = np.clip([pd_thr - pd_lo, pd_hi - pd_thr], 0, None)  # evita -1e-17 per Pd = 0 o 1
    ax2.errorbar(SNR_DB_LIST, pd_thr, yerr=yerr,
                 fmt='o-', capsize=4)
    ax2.set_xlabel("SNR a banda piena (dB)" if SNR_DEF == "fullband" else "SNR nella banda passante (dB)")
    ax2.set_ylabel("Pd")
    ax2.set_ylim(0, 1.02)
    ax2.set_title(f"Pd vs SNR, soglia {thr} (Pfa = {pfa_thr:.3f})")
    ax2.grid(True, ls='--', alpha=0.4)
    fig2.tight_layout()
    fig2.savefig(f"{OUT_PREFIX}_pd_vs_snr.pdf")

    np.savez(f"{OUT_PREFIX}_results.npz", snr_db=np.array(SNR_DB_LIST), T0=T0, T1=T1,
             threshold=thr, files=np.array(files))

    plt.show()


if __name__ == "__main__":
    main()