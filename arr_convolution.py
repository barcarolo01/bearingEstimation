"""
Pipeline: .arr (Bellhop/Bello) → Channel Impulse Response → convoluzione con .wav
Scenario: sorgente in superficie (barca) + idrofono sommerso, 1 TX / 1 RX
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import re
import sys
import os


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PARSER DEL FILE .arr
# ═══════════════════════════════════════════════════════════════════════════════

def parse_arr(filepath):
    """
    Legge un file .arr in formato Bello 2D con 1 TX / 1 RX.

    Struttura attesa:
        riga 1  : '2D'
        riga 2  : profondità sorgente
        riga 3  : n_sorgenti   scala
        riga 4  : n_punti_x    x0 x1 x2 ...   (griglia orizzontale)
        riga 5  : n_punti_z    z0 z1 z2 ...   (griglia verticale)
        riga 6  : n_raggi_totali
        poi, per ogni gruppo di arrivi:
            n_arrivi_gruppo          (intero)
            [n_arrivi_gruppo righe con: amp  angolo  tempo  slowness  x  z  f1  f2]
        Se n_arrivi_gruppo == 0, il gruppo è vuoto (salta).

    Restituisce:
        rays : lista di dict con chiavi
               'amplitude', 'angle', 'travel_time', 'slowness', 'x', 'z'
    """

    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    idx = 0

    # -- header --
    assert "'2D'" in lines[idx] or "2D" in lines[idx], "Formato non riconosciuto"
    idx += 1

    source_depth = float(lines[idx]); idx += 1
    n_src, scale  = lines[idx].split()[:2]
    n_src = int(n_src); idx += 1

    # griglia X
    tokens_x = lines[idx].split(); idx += 1
    n_x = int(tokens_x[0])
    x_grid = list(map(float, tokens_x[1:]))
    # la griglia può continuare su più righe
    while len(x_grid) < n_x:
        x_grid += list(map(float, lines[idx].split())); idx += 1

    # griglia Z
    tokens_z = lines[idx].split(); idx += 1
    n_z = int(tokens_z[0])
    z_grid = list(map(float, tokens_z[1:]))
    while len(z_grid) < n_z:
        z_grid += list(map(float, lines[idx].split())); idx += 1

    # numero totale di raggi attesi
    n_rays_total = int(lines[idx]); idx += 1

    rays = []

    while idx < len(lines):
        # quanti arrivi in questo gruppo?
        try:
            n_arrivals = int(lines[idx]); idx += 1
        except ValueError:
            idx += 1
            continue

        if n_arrivals == 0:
            continue

        for _ in range(n_arrivals):
            if idx >= len(lines):
                break
            parts = lines[idx].split(); idx += 1
            if len(parts) < 6:
                continue
            rays.append({
                'amplitude'  : float(parts[0]),
                'angle'      : float(parts[1]),
                'travel_time': float(parts[2]),
                'slowness'   : float(parts[3]),
                'x'          : float(parts[4]),
                'z'          : float(parts[5]),
            })

    print(f"[parse_arr] profondità sorgente: {source_depth} m")
    print(f"[parse_arr] raggi estratti: {len(rays)}")
    return rays, source_depth, x_grid, z_grid


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  COSTRUZIONE DELLA CHANNEL IMPULSE RESPONSE  h(t)
# ═══════════════════════════════════════════════════════════════════════════════

def build_cir(rays, fs, use_amplitudes=True, t_padding=0.05):
    """
    Costruisce la risposta impulsiva discreta del canale.

        h[n] = Σ_k  A_k · δ[n - round(τ_k · fs)]

    Parametri:
        rays           : lista di dict da parse_arr
        fs             : frequenza di campionamento [Hz]
        use_amplitudes : se False, tutti i contributi hanno ampiezza 1
                         (utile per vedere solo i tempi di arrivo)
        t_padding      : secondi di zero padding dopo l'ultimo arrivo

    Restituisce:
        h      : array numpy
        t_axis : asse temporale corrispondente [s]
    """
    travel_times = np.array([r['travel_time'] for r in rays])
    amplitudes   = np.array([r['amplitude']   for r in rays])

    t_max    = travel_times.max() + t_padding
    n_samples = int(t_max * fs) + 1
    h = np.zeros(n_samples)

    for A, tau in zip(amplitudes, travel_times):
        n = int(round(tau * fs))
        if 0 <= n < n_samples:
            h[n] += A if use_amplitudes else 1.0

    t_axis = np.arange(n_samples) / fs

    print(f"[build_cir] durata CIR: {t_max:.4f} s  ({n_samples} campioni @ {fs} Hz)")
    print(f"[build_cir] tempi arrivo: min={travel_times.min():.4f}s  "
          f"max={travel_times.max():.4f}s")
    print(f"[build_cir] ampiezze:    min={amplitudes.min():.3e}  "
          f"max={amplitudes.max():.3e}")

    return h, t_axis


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  CONVOLUZIONE CON IL FILE .wav
# ═══════════════════════════════════════════════════════════════════════════════

def convolve_with_wav(wav_path, h, output_path="segnale_idrofono.wav"):
    """
    Carica wav_path, convola con h, salva in output_path.
    Gestisce mono e stereo; normalizza l'output.
    """
    fs, x = wavfile.read(wav_path)

    # converti a float64 normalizzato
    if x.dtype == np.int16:
        x = x.astype(np.float64) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float64) / 2147483648.0
    elif x.dtype == np.uint8:
        x = (x.astype(np.float64) - 128) / 128.0
    else:
        x = x.astype(np.float64)

    # se stereo, usa solo il canale sinistro
    if x.ndim == 2:
        print("[wav] segnale stereo → uso solo il canale sinistro")
        x = x[:, 0]

    print(f"[wav] durata sorgente: {len(x)/fs:.3f} s  ({len(x)} campioni @ {fs} Hz)")

    # convoluzione
    y_full = fftconvolve(x, h, mode='full')
    fade_samples = len(h) - 1
    y = y_full[fade_samples : fade_samples + len(x)]

    # 3. Normalizzazione
    y /= np.max(np.abs(y))

    # salvataggio come float32
    wavfile.write(output_path, fs, y.astype(np.float32))
    print(f"[output] segnale idrofono salvato in: {output_path}")

    return fs, x, y


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  VISUALIZZAZIONE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(h, t_h, fs, x, y, rays, output_img="risultato.png"):

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Simulazione ricezione idrofono via ray tracing", fontsize=13, y=0.98)

    # ── pannello 1: CIR (stem plot) ──────────────────────────────────────────
    ax1 = fig.add_subplot(3, 1, 1)
    markerline, stemlines, baseline = ax1.stem(
        t_h * 1e3,          # ms
        h,
        markerfmt='C0o',
        linefmt='C0-',
        basefmt='k-'
    )
    plt.setp(stemlines, linewidth=0.8)
    plt.setp(markerline, markersize=4)
    ax1.set_title("Channel Impulse Response h(t)  —  arrivi dei raggi")
    ax1.set_xlabel("Tempo [ms]")
    ax1.set_ylabel("Ampiezza")
    ax1.grid(True, alpha=0.3)

    # annotazione: n° di raggi
    n_arrivals = sum(1 for v in h if v != 0)
    ax1.text(0.98, 0.92, f"{len(rays)} raggi totali\n{n_arrivals} campioni attivi",
             transform=ax1.transAxes, ha='right', va='top',
             fontsize=8, bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    # ── pannello 2: segnale sorgente ─────────────────────────────────────────
    ax2 = fig.add_subplot(3, 1, 2)
    t_x = np.arange(len(x)) / fs
    ax2.plot(t_x, x, color='C1', linewidth=0.6)
    ax2.set_title("Segnale sorgente x(t)  —  barca (input .wav)")
    ax2.set_xlabel("Tempo [s]")
    ax2.set_ylabel("Ampiezza norm.")
    ax2.grid(True, alpha=0.3)

    # ── pannello 3: segnale ricevuto ─────────────────────────────────────────
    ax3 = fig.add_subplot(3, 1, 3)
    t_y = np.arange(len(y)) / fs
    ax3.plot(t_y, y, color='C2', linewidth=0.6)
    ax3.set_title("Segnale ricevuto y(t)  —  idrofono simulato")
    ax3.set_xlabel("Tempo [s]")
    ax3.set_ylabel("Ampiezza norm.")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    print(f"[output] grafico salvato in: {output_img}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    input_names=["C:/Users/Nicola/Desktop/TESI/HYDRO2/hm_code/test_5_1/test_5.arr","C:/Users/Nicola/Desktop/TESI/HYDRO2/hm_code/test_5_2/test_5.arr"]
    outs_name=["1.wav","2.wav"]
    for i in range (2):
            
        # ── Configura qui i tuoi file ────────────────────────────────────────────
        ARR_FILE  = input_names[i]
        WAV_FILE  = "AudioFiles/0958_crop.wav"      # <-- il tuo file .wav (rumore barca)
        OUT_WAV   = outs_name[i]
        OUT_IMG   = "risultato.png"     # grafico
        # ────────────────────────────────────────────────────────────────────────

        # 1. parsing
        rays, src_depth, x_grid, z_grid = parse_arr(ARR_FILE)

        if len(rays) == 0:
            sys.exit("[ERRORE] Nessun raggio trovato. Controlla il file .arr")

        # 2. frequenza di campionamento dal .wav
        fs_wav, _ = wavfile.read(WAV_FILE)

        # 3. costruzione CIR
        h, t_h = build_cir(rays, fs=fs_wav, use_amplitudes=True)

        # 4. convoluzione
        fs, x, y = convolve_with_wav(WAV_FILE, h, output_path=OUT_WAV)

        # 5. visualizzazione
        #plot_results(h, t_h, fs, x, y, rays, output_img=OUT_IMG)
