"""
DEMON Spectrum Analyzer
=======================
Detection of Envelope Modulation On Noise

Analisi del DEMON spectrum per registrazioni di idrofoni.

Utilizzo:
    python demon_spectrum.py --file <percorso_audio> [opzioni]

Dipendenze:
    pip install numpy scipy matplotlib librosa
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt, hilbert, welch
import argparse
import os
import sys

# ──────────────────────────────────────────────
# Funzioni di supporto
# ──────────────────────────────────────────────

def load_audio(filepath):
    """
    Carica un file audio (WAV, FLAC, MP3, ecc.) usando librosa.
    Restituisce il segnale mono e il sample rate.
    """
    try:
        import librosa
        print(f"[*] Caricamento file: {filepath}")
        signal, sr = librosa.load(filepath, sr=None, mono=True)
        print(f"    Sample rate : {sr} Hz")
        print(f"    Durata      : {len(signal)/sr:.2f} s  ({len(signal)} campioni)")
        return signal, sr
    except ImportError:
        # Fallback: solo WAV tramite scipy
        from scipy.io import wavfile
        print(f"[*] Caricamento file WAV: {filepath}")
        sr, signal = wavfile.read(filepath)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        signal = signal.astype(np.float32)
        # Normalizzazione
        max_val = np.iinfo(np.int16).max if signal.dtype == np.int16 else 1.0
        signal = signal / max_val
        print(f"    Sample rate : {sr} Hz")
        print(f"    Durata      : {len(signal)/sr:.2f} s")
        return signal, sr


def bandpass_filter(signal, sr, f_low, f_high, order=5):
    """
    Applica un filtro Butterworth passa-banda al segnale.
    """
    nyq = sr / 2.0
    low  = f_low  / nyq
    high = f_high / nyq
    low  = np.clip(low,  1e-4, 0.9999)
    high = np.clip(high, 1e-4, 0.9999)
    if low >= high:
        raise ValueError(f"Banda non valida: f_low={f_low} Hz >= f_high={f_high} Hz")
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def extract_envelope(signal):
    """
    Estrae l'inviluppo del segnale tramite trasformata di Hilbert.
    """
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    return envelope


def decimate_envelope(envelope, sr, target_sr=1000):
    """
    Decimazione dell'inviluppo a target_sr Hz con anti-aliasing.
    Il DEMON spectrum ci interessa solo fino a ~300-500 Hz,
    quindi non serve lavorare a 96 kHz sull'inviluppo.
    """
    from scipy.signal import decimate
    factor = int(sr // target_sr)
    if factor <= 1:
        return envelope, sr
    # decimate applica internamente un filtro anti-aliasing
    env_dec = envelope.copy()
    # Decomposizione in passi da al più 13 per stabilità numerica
    while factor > 1:
        step = min(factor, 10)
        env_dec = decimate(env_dec, step, ftype='fir', zero_phase=True)
        factor = factor // step
    actual_sr = len(env_dec) / (len(envelope) / sr)
    return env_dec.astype(np.float32), actual_sr


def highpass_envelope(envelope, sr, f_cutoff=2.0, order=4):
    """
    Filtro passa-alto sull'inviluppo decimato.
    Elimina modulazioni lente (moto ondoso, rollio, variazioni di rotta)
    prima del calcolo del DEMON spectrum.
    """
    nyq = sr / 2.0
    wn  = np.clip(f_cutoff / nyq, 1e-4, 0.9999)
    b, a = butter(order, wn, btype='high')
    return filtfilt(b, a, envelope)


def compute_demon_spectrum(envelope, sr, nperseg=None, noverlap=None, nfft=None):
    """
    Calcola il DEMON spectrum tramite Welch method sull'inviluppo.
    Restituisce frequenze (Hz) e densità spettrale di potenza (dB).
    
    nperseg grande → alta risoluzione frequenziale (es. 0.1 Hz con sr=1000 e nperseg=10000)
    """
    if nperseg is None:
        # Mira a ~0.1 Hz di risoluzione: nperseg = sr / df
        nperseg = min(len(envelope), int(sr / 0.1))
        nperseg = max(nperseg, 1024)
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        # Zero-padding x2 per interpolazione spettrale
        nfft = nperseg * 2

    freqs, psd = welch(envelope, fs=sr, nperseg=nperseg,
                       noverlap=noverlap, nfft=nfft,
                       window='hann', scaling='density')

    psd_db = 10 * np.log10(psd + 1e-20)
    return freqs, psd_db


def find_peaks_demon(freqs, psd_db, n_peaks=5, min_freq=0.3):
    """
    Individua i picchi principali nel DEMON spectrum.
    Usa una prominenza adattiva basata sulla mediana (più robusta al rumore).
    """
    from scipy.signal import find_peaks

    mask = freqs >= min_freq
    f_masked   = freqs[mask]
    psd_masked = psd_db[mask]

    # Prominenza adattiva: 1.5 dB sopra la mediana locale è già significativo
    median_psd = np.median(psd_masked)
    noise_floor = np.percentile(psd_masked, 25)
    prominence = max(1.5, (median_psd - noise_floor) * 0.3)

    # distance minima = ~0.5 Hz in bin
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    min_distance = max(3, int(0.5 / df))

    peaks, props = find_peaks(psd_masked, prominence=prominence,
                               distance=min_distance)

    if len(peaks) > 0:
        order = np.argsort(psd_masked[peaks])[::-1]
        peaks = peaks[order[:n_peaks]]
        return f_masked[peaks], psd_masked[peaks]
    return np.array([]), np.array([])


# ──────────────────────────────────────────────
# Visualizzazione
# ──────────────────────────────────────────────

def plot_demon_analysis(signal, sr, envelope, freqs, psd_db,
                        f_low, f_high, peak_freqs, peak_amps,
                        filename="", envelope_sr=None):
    """
    Produce una figura a 4 pannelli:
      1. Forma d'onda originale
      2. Segnale filtrato + inviluppo
      3. DEMON spectrum completo
      4. Zoom DEMON spectrum sulle basse frequenze
    """
    t = np.arange(len(signal)) / sr

    fig = plt.figure(figsize=(14, 10), facecolor="#0d1117")
    fig.suptitle(
        f"DEMON Spectrum Analysis\n{os.path.basename(filename) if filename else 'Hydrophone Signal'}",
        color="#e6edf3", fontsize=15, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.55)

    ax_color   = "#e6edf3"
    ax_bg      = "#161b22"
    grid_color = "#30363d"
    accent1    = "#58a6ff"   # blu – forma d'onda
    accent2    = "#3fb950"   # verde – inviluppo
    accent3    = "#ff7b72"   # rosso – picchi

    def _style(ax, title):
        ax.set_facecolor(ax_bg)
        ax.spines[:].set_color(grid_color)
        ax.tick_params(colors=ax_color, labelsize=8)
        ax.xaxis.label.set_color(ax_color)
        ax.yaxis.label.set_color(ax_color)
        ax.set_title(title, color=ax_color, fontsize=10, pad=6)
        ax.grid(True, color=grid_color, linewidth=0.5, linestyle="--", alpha=0.7)

    # ── Pannello 1: forma d'onda originale ──
    ax1 = fig.add_subplot(gs[0])
    # Decimazione per velocità di rendering
    dec = max(1, len(t) // 8000)
    ax1.plot(t[::dec], signal[::dec], color=accent1, linewidth=0.6, alpha=0.85)
    ax1.set_xlabel("Tempo (s)")
    ax1.set_ylabel("Ampiezza")
    _style(ax1, "① Forma d'onda originale")

    # ── Pannello 2: segnale filtrato + inviluppo (decimato) ──
    ax2 = fig.add_subplot(gs[1])
    filtered = bandpass_filter(signal, sr, f_low, f_high)
    t_env = np.arange(len(envelope)) / (envelope_sr if envelope_sr else sr)
    ax2.plot(t[::dec], filtered[::dec], color=accent1, linewidth=0.5,
             alpha=0.5, label="Segnale filtrato")
    ax2.plot(t_env[::max(1, len(t_env)//8000)],
             envelope[::max(1, len(t_env)//8000)],
             color=accent2, linewidth=1.0, alpha=0.9, label="Inviluppo (decimato)")
    ax2.set_xlabel("Tempo (s)")
    ax2.set_ylabel("Ampiezza")
    ax2.legend(fontsize=8, facecolor=ax_bg, edgecolor=grid_color,
               labelcolor=ax_color, loc="upper right")
    _style(ax2, f"② Segnale filtrato ({f_low:.0f}–{f_high:.0f} Hz) + Inviluppo")

    # ── Pannello 3: DEMON spectrum completo ──
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(freqs, psd_db, color=accent1, linewidth=0.8)
    if len(peak_freqs) > 0:
        ax3.scatter(peak_freqs, peak_amps, color=accent3, zorder=5,
                    s=50, label="Picchi principali")
        for f, a in zip(peak_freqs, peak_amps):
            ax3.annotate(f"{f:.2f} Hz", xy=(f, a),
                         xytext=(5, 5), textcoords="offset points",
                         fontsize=7, color=accent3)
        ax3.legend(fontsize=8, facecolor=ax_bg, edgecolor=grid_color,
                   labelcolor=ax_color)
    ax3.set_xlabel("Frequenza (Hz)")
    ax3.set_ylabel("PSD (dB)")
    _style(ax3, "③ DEMON Spectrum (spettro dell'inviluppo)")

    # ── Pannello 4: zoom basse frequenze (0–300 Hz) ──
    ax4 = fig.add_subplot(gs[3])
    zoom_max = min(100.0, freqs[-1])
    mask = freqs <= zoom_max
    ax4.plot(freqs[mask], psd_db[mask], color=accent2, linewidth=0.9)
    if len(peak_freqs) > 0:
        zoom_peaks = peak_freqs[peak_freqs <= zoom_max]
        zoom_amps  = peak_amps[peak_freqs  <= zoom_max]
        if len(zoom_peaks) > 0:
            ax4.scatter(zoom_peaks, zoom_amps, color=accent3, zorder=5, s=50)
            for f, a in zip(zoom_peaks, zoom_amps):
                ax4.annotate(f"{f:.2f} Hz", xy=(f, a),
                             xytext=(5, 5), textcoords="offset points",
                             fontsize=7, color=accent3)
    ax4.set_xlabel("Frequenza (Hz)")
    ax4.set_ylabel("PSD (dB)")
    ax4.set_xlim(0, zoom_max)
    _style(ax4, f"④ DEMON Spectrum – Zoom 0–{zoom_max:.0f} Hz (shaft/blade rate)")

    plt.savefig("demon_spectrum_output.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("[*] Figura salvata: demon_spectrum_output.png")
    plt.show()


# ──────────────────────────────────────────────
# Generatore di segnale sintetico (demo)
# ──────────────────────────────────────────────

def generate_synthetic_hydrophone(duration=30.0, sr=96000,
                                  shaft_rate=6.5, n_blades=4):
    """
    Genera un segnale sintetico che simula il rumore di un'elica:
    - Rumore a banda larga (oceano + cavitazione)
    - Modulazione d'ampiezza a shaft rate e blade rate
    - Armoniche di blade rate
    """
    t = np.arange(int(duration * sr)) / sr
    blade_rate = shaft_rate * n_blades

    # Rumore bianco → colorato con filtro passa-banda
    noise = np.random.randn(len(t))
    b, a  = butter(4, [2000/sr*2, 20000/sr*2], btype='band')
    noise = filtfilt(b, a, noise)

    # Modulazione: shaft + blade + 2° armonica blade
    mod  = 1.0
    mod += 0.40 * np.sin(2 * np.pi * shaft_rate  * t)
    mod += 0.60 * np.sin(2 * np.pi * blade_rate  * t + 0.3)
    mod += 0.25 * np.sin(2 * np.pi * blade_rate * 2 * t)
    mod  = (mod - mod.min()) / (mod.max() - mod.min() + 1e-9)

    signal = noise * mod
    signal += 0.05 * np.random.randn(len(t))   # rumore di fondo
    signal /= np.abs(signal).max()

    print(f"[*] Segnale sintetico generato")
    print(f"    Shaft rate : {shaft_rate} Hz")
    print(f"    Blade rate : {blade_rate} Hz  ({n_blades} pale)")
    return signal.astype(np.float32), sr


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DEMON Spectrum Analyzer per registrazioni di idrofono"
    )
    parser.add_argument("--file",    type=str,   default=None,
                        help="Percorso del file audio (WAV, FLAC, MP3, …)")
    parser.add_argument("--f_low",   type=float, default=2000.0,
                        help="Freq. inferiore del filtro passa-banda (Hz) [default: 2000]")
    parser.add_argument("--f_high",  type=float, default=20000.0,
                        help="Freq. superiore del filtro passa-banda (Hz) [default: 20000]")
    parser.add_argument("--nperseg", type=int,   default=None,
                        help="Campioni per segmento FFT Welch [default: auto (sr/0.1)]")
    parser.add_argument("--hp_freq",  type=float, default=None,
                        help="Freq. di taglio passa-alto sull'inviluppo (Hz) [default: disabilitato]. "
                             "Es: --hp_freq 2.0 per eliminare picchi sotto 2 Hz (moto ondoso)")
    parser.add_argument("--dec_sr",  type=int,   default=1000,
                        help="Sample rate target dopo decimazione inviluppo (Hz) [default: 1000]")
    parser.add_argument("--n_peaks", type=int,   default=3,
                        help="Numero di picchi da evidenziare [default: 6]")
    parser.add_argument("--demo",    action="store_true",
                        help="Usa segnale sintetico di esempio (non serve file)")
    args = parser.parse_args()

    # ── Caricamento / generazione ──
    if args.demo or args.file is None:
        if args.file is None:
            print("[!] Nessun file specificato – uso modalità DEMO (segnale sintetico).")
        signal, sr = generate_synthetic_hydrophone()
        filename = "demo_synthetic"
    else:
        signal, sr = load_audio(args.file)
        filename = args.file

    # ── Pipeline DEMON ──
    print(f"\n[*] Pipeline DEMON")
    print(f"    1. Filtro passa-banda: {args.f_low}–{args.f_high} Hz")
    filtered = bandpass_filter(signal, sr, args.f_low, args.f_high)

    print(f"    2. Estrazione inviluppo (Hilbert)")
    envelope = extract_envelope(filtered)

    print(f"    3. Decimazione inviluppo a ~{args.dec_sr} Hz (anti-aliasing)")
    envelope_dec, env_sr = decimate_envelope(envelope, sr, target_sr=args.dec_sr)
    print(f"       SR effettivo post-decimazione: {env_sr:.1f} Hz")

    if args.hp_freq is not None:
        print(f"    3b. Filtro passa-alto sull'inviluppo: {args.hp_freq} Hz (rimozione moto ondoso)")
        envelope_dec = highpass_envelope(envelope_dec, env_sr, f_cutoff=args.hp_freq)

    print(f"    4. Calcolo spettro Welch dell'inviluppo (nperseg={'auto' if args.nperseg is None else args.nperseg})")
    freqs, psd_db = compute_demon_spectrum(envelope_dec, env_sr, nperseg=args.nperseg)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 0
    print(f"       Risoluzione frequenziale: {df:.4f} Hz/bin")

    print(f"    5. Rilevamento picchi")
    peak_freqs, peak_amps = find_peaks_demon(freqs, psd_db, n_peaks=args.n_peaks)

    if len(peak_freqs) > 0:
        print(f"\n[*] Picchi rilevati nel DEMON spectrum:")
        for f, a in sorted(zip(peak_freqs, peak_amps)):
            print(f"    {f:7.3f} Hz  →  {a:.1f} dB")

    # ── Visualizzazione ──
    print("\n[*] Generazione grafico …")
    plot_demon_analysis(signal, sr, envelope_dec, freqs, psd_db,
                        args.f_low, args.f_high,
                        peak_freqs, peak_amps,
                        filename=filename, envelope_sr=env_sr)


if __name__ == "__main__":
    main()
