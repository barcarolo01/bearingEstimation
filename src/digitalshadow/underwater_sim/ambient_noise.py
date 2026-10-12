"""
Rumore ambientale (modello di Wenz) da aggiungere alle tracce simulate.

Tutti i segnali sono pressione in Pa (tracce calibrate con sl_db).
Il rumore e' colorato con la PSD di Wenz su tutta la banda [0, fs/2] ed e'
spazialmente coerente tra idrofoni secondo il modello di rumore diffuso isotropo:
coerenza tra due idrofoni a distanza d = sinc(2 f d / c). Vuol dire che sotto
~c/(2d) (2.5 kHz con d = 30 cm) il rumore e' in parte uguale sugli idrofoni.

Configurazione tramite variabili d'ambiente (lette a ogni chiamata):
    ADD_AMBIENT_NOISE   1/0, true/false   flag globale (default: 0 = disattivato)
    NOISE_SHIPPING      0 ... 1           attivita' di traffico navale (default 0.5)
    NOISE_WIND          m/s               velocita' del vento (default 5)
    NOISE_EXTRA_DB      dB                offset su tutta la PSD, es. +5..10 per sito costiero (default 0)
    NOISE_SPACING       m                 distanza tra idrofoni adiacenti, array lineare (default 0.3)
    NOISE_COHERENT      1/0, true/false   rumore coerente tra idrofoni (default 1); 0 = indipendente
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
        extra_db=_env_float("NOISE_EXTRA_DB", 5.0),
        spacing=_env_float("NOISE_SPACING", 0.3),
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


def expected_noise_db(fs, shipping=0.5, wind=5.0, extra_db=0.0):
    """Livello di rumore atteso integrato su tutta la banda [1 Hz, fs/2], dB re 1 µPa."""
    # griglia logaritmica: le basse frequenze dominano l'integrale
    f = np.geomspace(1.0, fs / 2, 4000)
    _trapz = getattr(np, "trapezoid", None) or np.trapz      # numpy 2.x / 1.x
    return 10 * np.log10(_trapz(10 ** (wenz_psd_db(f, shipping, wind) / 10), f)) + extra_db


def generate_noise(n_samples, fs, shipping=0.5, wind=5.0, rng=None, extra_db=0.0):
    """
    Rumore [Pa] con PSD monolatera pari al modello di Wenz su [0, fs/2].

    Sintesi a fase casuale: ogni bin ha ampiezza esattamente pari alla PSD
    del modello e fase uniforme indipendente. Le realizzazioni sono scorrelate
    tra loro (e gaussiane nel tempo per il teorema del limite centrale), ma la
    potenza totale e' la stessa per tutte: niente fluttuazioni di NL tra tracce
    dovute ai pochi bin a bassa frequenza che dominano la potenza a banda piena.
    """
    rng = np.random.default_rng() if rng is None else rng
    f = rfftfreq(n_samples, 1 / fs)
    psd_pa2 = (P_REF ** 2) * 10 ** ((wenz_psd_db(f, shipping, wind) + extra_db) / 10)
    # |X_k|^2 = N * PSD * fs / 2  ->  potenza per bin = PSD * df (Parseval)
    amp = np.sqrt(n_samples * psd_pa2 * fs / 2)
    amp[f < 1.0] = 0.0                                 # niente DC ne' bin < 1 Hz (fuori dal modello)
    phase = rng.uniform(0.0, 2 * np.pi, len(f))
    if n_samples % 2 == 0:                             # bin di Nyquist: deve essere reale
        phase[-1] = 0.0
    return irfft(amp * np.exp(1j * phase), n=n_samples)


def generate_noise_array(n_samples, fs, n_ch, spacing=0.3, positions=None, c=1500.0,
                         shipping=0.5, wind=5.0, rng=None, extra_db=0.0, coherent=True):
    """
    Rumore di Wenz [Pa] per n_ch idrofoni, coerente come un campo diffuso isotropo.

    positions: array (n_ch, 3) in metri. Se None, array lineare con passo `spacing`.
    Restituisce un array (n_ch, n_samples). Ogni canale ha la PSD di Wenz;
    la coerenza tra i canali i, j alla frequenza f e' sinc(2 f d_ij / c).
    """
    rng = np.random.default_rng() if rng is None else rng
    if positions is None:
        positions = np.zeros((n_ch, 3))
        positions[:, 0] = np.arange(n_ch) * spacing
    positions = np.asarray(positions, dtype=float)
    dist = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)

    f = rfftfreq(n_samples, 1 / fs)
    psd_pa2 = (P_REF ** 2) * 10 ** ((wenz_psd_db(f, shipping, wind) + extra_db) / 10)
    amp = np.sqrt(n_samples * psd_pa2 * fs / 2)
    amp[f < 1.0] = 0.0

    # matrice di coerenza per ogni bin e sua "radice" L, con L L^T = coerenza
    if coherent:
        coh = np.sinc(2 * f[:, None, None] * dist[None] / c)      # (n_bin, n_ch, n_ch)
    else:
        coh = np.broadcast_to(np.eye(n_ch), (len(f), n_ch, n_ch)) # canali indipendenti
    w, v = np.linalg.eigh(coh)
    L = v * np.sqrt(np.clip(w, 0, None))[:, None, :]

    # fasi casuali indipendenti, poi mescolate tra canali secondo la coerenza
    u = np.exp(1j * rng.uniform(0, 2 * np.pi, (len(f), n_ch)))
    if n_samples % 2 == 0:
        u[-1] = 1.0                                                # Nyquist reale
    X = amp[:, None] * np.einsum("kij,kj->ki", L, u)
    if n_samples % 2 == 0:
        X[-1] = X[-1].real
    return irfft(X, n=n_samples, axis=0).T


# ---------------------------------------------------------------- utilita'
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
                      spacing=None, positions=None, c=1500.0, coherent=None,
                      labels=None, verbose=True):
    """
    Aggiunge rumore di Wenz a banda piena, spazialmente coerente, alle tracce pulite [Pa].

    spacing:   distanza tra idrofoni adiacenti [m], array lineare
               (None = variabile NOISE_SPACING, default 0.3 m).
    positions: posizioni (n_ch, 3) in metri; se date, hanno la precedenza su spacing.
    Tutte le tracce devono avere la stessa lunghezza.

    Restituisce (tracce_rumorose, snr_dB).
    L'SNR e' a banda piena [0, fs/2], con la potenza del segnale
    misurata solo dove il segnale e' presente.
    """
    rng = np.random.default_rng() if rng is None else rng
    labels = labels or [f"H{i + 1}" for i in range(len(signals))]
    if spacing is None:
        spacing = _env_float("NOISE_SPACING", 0.3)
    if coherent is None:
        coherent = False
    tiny = 1e-30

    n = len(signals[0])
    if any(len(s) != n for s in signals):
        raise ValueError("Tutte le tracce devono avere la stessa lunghezza.")
    noises = generate_noise_array(n, fs, len(signals), spacing, positions, c,
                                  shipping, wind, rng, extra_db, coherent)

    noisy, snrs = [], []
    for sig, noise, lab in zip(signals, noises, labels):
        sig = np.asarray(sig, dtype=float)

        ps = _active_power(sig, fs)
        pn = np.mean(noise ** 2)
        rl_db = 10 * np.log10(max(ps, tiny) / P_REF ** 2)
        nl_db = 10 * np.log10(max(pn, tiny) / P_REF ** 2)
        snr = rl_db - nl_db

        if verbose:
            print(f"[{lab:>4}] RL {rl_db:6.1f} | NL {nl_db:6.1f} "
                  f"dB re 1 µPa | SNR (banda piena) {snr:+6.1f} dB")
        noisy.append(sig + noise)
        snrs.append(snr)
    return noisy, snrs