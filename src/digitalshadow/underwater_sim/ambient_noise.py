"""
Rumore ambientale (modello di Wenz) da aggiungere alle tracce simulate.

Tutti i segnali sono pressione in Pa (tracce calibrate con sl_db).
Il rumore e' gaussiano, colorato con la PSD di Wenz e indipendente tra idrofoni.

Configurazione tramite variabili d'ambiente (lette a ogni chiamata):
    ADD_AMBIENT_NOISE   1/0, true/false   flag globale (default: 0 = disattivato)
    NOISE_SHIPPING      0 ... 1           attivita' di traffico navale (default 0.5)
    NOISE_WIND          m/s               velocita' del vento (default 5)
    NOISE_EXTRA_DB      dB                offset su tutta la PSD, es. +5..10 per sito costiero (default 0)
    SIGNAL_F_LOW        Hz                inizio banda del segnale (default 5000)
    SIGNAL_F_HIGH       Hz                fine banda del segnale (default 18000)
    NOISE_SEED          intero            seed per riproducibilita' (default: nessuno = casuale)
"""
import os
import numpy as np
from scipy.fft import rfft, irfft, rfftfreq

P_REF = 1e-6  # Pa (1 µPa)

# ---------------------------------------------------------------- configurazione
def _env_bool(name, default=False):
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name, default):
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else float(v)


def get_noise_config():
    """Legge la configurazione del rumore dalle variabili d'ambiente."""
    seed = os.getenv("NOISE_SEED")
    return dict(
        enabled=_env_bool("ADD_AMBIENT_NOISE", False),
        shipping=_env_float("NOISE_SHIPPING", 0.5),
        wind=_env_float("NOISE_WIND", 5.0),
        extra_db=_env_float("NOISE_EXTRA_DB", 5.0), #HERE
        f_low=_env_float("SIGNAL_F_LOW", 5000.0),
        f_high=_env_float("SIGNAL_F_HIGH", 18000.0),
        seed=int(seed) if seed not in (None, "") else None,
    )


# ---------------------------------------------------------------- modello di Wenz
def wenz_psd_db(f_hz, shipping=0.5, wind=5.0):
    """
    PSD del rumore ambientale, dB re 1 µPa^2/Hz (parametrizzazione Coates / Stojanovic).
    f_hz: frequenze in Hz; shipping: 0..1; wind: m/s.
    """
    f = np.maximum(np.asarray(f_hz, dtype=float), 1.0) / 1000.0   # kHz, evita log(0)
    n_turb = 17 - 30 * np.log10(f)
    n_ship = 40 + 20 * (shipping - 0.5) + 26 * np.log10(f) - 60 * np.log10(f + 0.03)
    n_wind = 50 + 7.5 * np.sqrt(wind) + 20 * np.log10(f) - 40 * np.log10(f + 0.4)
    n_th = -15 + 20 * np.log10(f)
    total = sum(10 ** (n / 10) for n in (n_turb, n_ship, n_wind, n_th))
    return 10 * np.log10(total)


def expected_band_noise_db(f_low, f_high, shipping=0.5, wind=5.0, extra_db=0.0):
    """Livello di rumore atteso integrato in [f_low, f_high], dB re 1 µPa."""
    f = np.linspace(f_low, f_high, 4000)
    _trapz = getattr(np, "trapezoid", None) or np.trapz      # numpy 2.x / 1.x
    return 10 * np.log10(_trapz(10 ** (wenz_psd_db(f, shipping, wind) / 10), f)) + extra_db


def generate_noise(n_samples, fs, shipping=0.5, wind=5.0, rng=None, extra_db=0.0):
    """Rumore gaussiano [Pa] con PSD monolatera pari al modello di Wenz."""
    rng = np.random.default_rng() if rng is None else rng
    w = rng.standard_normal(n_samples)                 # bianco, varianza 1: PSD = 2/fs
    f = rfftfreq(n_samples, 1 / fs)
    psd_pa2 = (P_REF ** 2) * 10 ** ((wenz_psd_db(f, shipping, wind) + extra_db) / 10)
    shaping = np.sqrt(psd_pa2 * fs / 2)
    shaping[0] = 0.0                                   # niente componente continua
    return irfft(rfft(w) * shaping, n=n_samples)


# ---------------------------------------------------------------- utilita'
def _bandpass(x, fs, f_low, f_high):
    X = rfft(x)
    f = rfftfreq(len(x), 1 / fs)
    X[(f < f_low) | (f > f_high)] = 0
    return irfft(X, n=len(x))


def _active_power(x, fs, win_s=0.05, thr_db=-20.0):
    """Potenza media [Pa^2] solo sulle finestre in cui c'e' segnale."""
    n = max(1, int(win_s * fs))
    n_win = len(x) // n
    if n_win == 0:
        return np.mean(x ** 2)
    p = np.mean(x[:n_win * n].reshape(n_win, n) ** 2, axis=1)
    if p.max() <= 0:
        return 0.0
    return np.mean(p[p > p.max() * 10 ** (thr_db / 10)])


# ---------------------------------------------------------------- funzione principale
def add_ambient_noise(signals, fs, shipping=0.5, wind=5.0, extra_db=0.0, rng=None,
                      f_low=5000.0, f_high=18000.0, labels=None, verbose=True):
    """
    Aggiunge un rumore di Wenz indipendente a ogni traccia pulita [Pa].

    Restituisce (tracce_rumorose, snr_in_banda_dB).
    L'SNR e' calcolato in [f_low, f_high], con la potenza del segnale
    misurata solo dove il segnale e' presente.
    """
    rng = np.random.default_rng() if rng is None else rng
    labels = labels or [f"H{i + 1}" for i in range(len(signals))]
    tiny = 1e-30

    noisy, snrs = [], []
    for sig, lab in zip(signals, labels):
        sig = np.asarray(sig, dtype=float)
        noise = generate_noise(len(sig), fs, shipping, wind, rng, extra_db)

        ps = _active_power(_bandpass(sig, fs, f_low, f_high), fs)
        pn = np.mean(_bandpass(noise, fs, f_low, f_high) ** 2)
        rl_db = 10 * np.log10(max(ps, tiny) / P_REF ** 2)
        nl_db = 10 * np.log10(max(pn, tiny) / P_REF ** 2)
        snr = rl_db - nl_db

        if verbose:
            print(f"[{lab:>4}] RL in banda {rl_db:6.1f} | NL in banda {nl_db:6.1f} "
                  f"dB re 1 µPa | SNR in banda {snr:+6.1f} dB")
        noisy.append(sig + noise)
        snrs.append(snr)
    return noisy, snrs
