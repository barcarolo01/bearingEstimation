"""
bellhop_to_wav.py
─────────────────────────────────────────────────────────────────────
Genera file .wav simulati per RX1, RX2 (e opzionalmente RX3)
a partire da file .arr di Bellhop.

UTILIZZO:
    # Sorgente sintetica, 1 arrivo per RD (default):
    python bellhop_to_wav.py --rx1 RX1.arr --rx2 RX2.arr

    # Con terzo ricevitore:
    python bellhop_to_wav.py --rx1 RX1.arr --rx2 RX2.arr --rx3 RX3.arr

    # Sorgente da file audio, 10 arrivi per RD:
    python bellhop_to_wav.py --rx1 RX1.arr --rx2 RX2.arr --source audio.wav --n_arrivals 10

    # Tutti gli arrivi:
    python bellhop_to_wav.py --rx1 RX1.arr --rx2 RX2.arr --n_arrivals 0

DIPENDENZE:
    pip install numpy scipy soundfile
"""

import argparse
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve, correlate, resample_poly
from math import gcd
import soundfile as sf
import os

# ═══════════════════════════════════════════════════════════════════════
# PARAMETRI CONFIGURABILI
# ═══════════════════════════════════════════════════════════════════════

FS_OUT       = 96000     # Hz — frequenza di campionamento output
DURATION     = 5.0       # secondi — durata sorgente sintetica
FREQ_CENTER  = 34.0      # Hz — frequenza fondamentale motore (sorgente sintetica)
BW_LOW       = 5.0       # Hz — bordo basso banda (sorgente sintetica)
BW_HIGH      = 200.0     # Hz — bordo alto banda (sorgente sintetica)
SOUND_SPEED  = 1526.85   # m/s — velocità del suono (per verifica teorica)
SEED         = 42        # seed random per riproducibilità

# ═══════════════════════════════════════════════════════════════════════


def read_arr(filename):
    with open(filename) as f:
        lines = [l.strip() for l in f.readlines()]

    i = 0
    i += 1  # '2D'
    i += 1  # frequenza
    parts = lines[i].split(); i += 1  # NSD + SD

    rd_parts = lines[i].split()
    nrd = int(rd_parts[0])
    rd_values = list(map(float, rd_parts[1:])); i += 1
    while len(rd_values) < nrd:
        rd_values += list(map(float, lines[i].split())); i += 1

    rr_parts = lines[i].split()
    nr = int(rr_parts[0])
    rr_values = list(map(float, rr_parts[1:])); i += 1
    while len(rr_values) < nr:
        rr_values += list(map(float, lines[i].split())); i += 1

    i += 1  # salta intero header globale

    arrivals = {}
    for rd in rd_values:
        for rr in rr_values:
            narr = int(lines[i]); i += 1
            arr_list = []
            for _ in range(narr):
                p = lines[i].split(); i += 1
                arr_list.append((float(p[0]), float(p[1]), float(p[2])))
            arrivals[(round(rd, 6), round(rr, 3))] = arr_list

    return rr_values, rd_values, arrivals


def make_ship_noise(duration, fs, f_center, bw_low, bw_high, seed=42):
    np.random.seed(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    noise = np.random.randn(n)
    sos = butter(4, [bw_low, bw_high], btype='bandpass', fs=fs, output='sos')
    noise_filt = sosfilt(sos, noise)
    motor = np.zeros(n)
    for k, amp in enumerate([1.0, 0.6, 0.4, 0.25, 0.15], start=1):
        motor += amp * np.sin(2 * np.pi * f_center * k * t)
    envelope = 0.5 + 0.5 * np.abs(motor / np.max(np.abs(motor)))
    signal = noise_filt * envelope + 0.3 * motor
    signal /= np.max(np.abs(signal))
    return signal.astype(np.float32)


def load_audio_source(filepath, fs_target):
    data, fs_orig = sf.read(filepath, dtype='float32')
    if data.ndim > 1:
        data = data.mean(axis=1)
        print(f"  Convertito da stereo a mono")
    if fs_orig != fs_target:
        g = gcd(fs_target, fs_orig)
        up = fs_target // g
        down = fs_orig // g
        data = resample_poly(data, up, down).astype(np.float32)
        print(f"  Ricampionato da {fs_orig} Hz a {fs_target} Hz")
    else:
        print(f"  Frequenza di campionamento: {fs_orig} Hz (nessun resample)")
    data /= np.max(np.abs(data))
    print(f"  Durata: {len(data)/fs_target:.2f} s ({len(data)} campioni)")
    return data


def build_ir(arrivals_dict, rd_values, rr_target, fs, n_arrivals=1):
    """
    Costruisce la risposta impulsiva dagli arrivi Bellhop.

    Per ogni RD, seleziona i primi N arrivi in ordine temporale
    e li somma nella IR come spike (ampiezza * cos(fase)) al loro
    campione corrispondente.

    Parametri
    ─────────
    arrivals_dict : dict {(rd, rr): [(amp, phase, time), ...]}
    rd_values     : lista delle profondità ricevitori
    rr_target     : range del ricevitore di interesse (m)
    fs            : frequenza di campionamento (Hz)
    n_arrivals    : numero di arrivi da usare per RD, ordinati per tempo.
                    0 = usa tutti gli arrivi disponibili.

    Ritorna
    ───────
    h             : risposta impulsiva (float32)
    used_arrivals : lista totale di (amp, phase, time) inseriti nella IR
    """
    used_arrivals = []

    for rd in rd_values:
        key = (round(rd, 6), round(rr_target, 3))
        arr_list = arrivals_dict.get(key, [])
        if not arr_list:
            continue

        # Ordina per tempo crescente
        sorted_arr = sorted(arr_list, key=lambda a: a[2])

        # Seleziona i primi N (0 = tutti)
        if n_arrivals > 0:
            selected = sorted_arr[:n_arrivals]
        else:
            selected = sorted_arr

        used_arrivals.extend(selected)

    if not used_arrivals:
        print("  ⚠️  Nessun arrivo trovato per questo range!")
        return np.zeros(int(0.01 * fs), dtype=np.float32), []

    max_time = max(a[2] for a in used_arrivals)
    n = int(max_time * fs) + int(0.05 * fs)
    h = np.zeros(n, dtype=np.float64)

    for amp, phase, time in used_arrivals:
        sample = int(round(time * fs))
        if 0 <= sample < n:
            h[sample] += amp * np.cos(phase)

    return h.astype(np.float32), used_arrivals


def main():
    parser = argparse.ArgumentParser(
        description="Genera .wav simulati per RX1/RX2 da file .arr di Bellhop"
    )
    parser.add_argument('--rx1',        required=True,       help='File .arr per RX1')
    parser.add_argument('--rx2',        required=True,       help='File .arr per RX2')
    parser.add_argument('--rx3',        default=None,        help='File .arr per RX3 (opzionale)')
    parser.add_argument('--source',     default=None,        help='File audio sorgente (opzionale)')
    parser.add_argument('--out1',       default='RX1_received.wav')
    parser.add_argument('--out2',       default='RX2_received.wav')
    parser.add_argument('--out3',       default='RX3_received.wav')
    parser.add_argument('--fs',         default=FS_OUT,      type=int,   help=f'Sample rate output (default: {FS_OUT})')
    parser.add_argument('--dur',        default=DURATION,    type=float, help=f'Durata sorgente sintetica in s (default: {DURATION})')
    parser.add_argument('--fc',         default=FREQ_CENTER, type=float, help=f'Freq. motore Hz (default: {FREQ_CENTER})')
    parser.add_argument('--n_arrivals', default=1,           type=int,
                        help='Arrivi per RD ordinati per tempo (default: 1). '
                             '0 = tutti. Valori consigliati: 1 (pulito), 5-15 (realistico).')
    args = parser.parse_args()

    fs = args.fs
    n_arr = args.n_arrivals
    label = str(n_arr) if n_arr > 0 else "tutti"

    # ── Sorgente ──────────────────────────────────────────────────────
    if args.source:
        print(f"\n📂 Caricamento sorgente: {args.source}")
        source = load_audio_source(args.source, fs)
    else:
        print(f"\n🔊 Generazione sorgente sintetica (nave, {args.fc} Hz)...")
        source = make_ship_noise(args.dur, fs, args.fc, BW_LOW, BW_HIGH, seed=SEED)
        print(f"  Durata: {args.dur} s ({len(source)} campioni) @ {fs} Hz")

    # ── Lettura .arr ──────────────────────────────────────────────────
    print(f"\n📖 Lettura {args.rx1}...")
    rr1_vals, rd1_vals, arr1 = read_arr(args.rx1)
    rr_rx1 = max(rr1_vals)

    print(f"📖 Lettura {args.rx2}...")
    rr2_vals, rd2_vals, arr2 = read_arr(args.rx2)
    rr_rx2 = max(rr2_vals)

    if args.rx3:
        print(f"📖 Lettura {args.rx3}...")
        rr3_vals, rd3_vals, arr3 = read_arr(args.rx3)
        rr_rx3 = max(rr3_vals)

    # ── Risposta impulsiva ─────────────────────────────────────────────
    print(f"\n⚙️  IR per RX1 — range {rr_rx1:.4f} m  [{label} arrivi per RD]")
    h1, used1 = build_ir(arr1, rd1_vals, rr_rx1, fs, n_arrivals=n_arr)
    print(f"  Spike totali nella IR: {len(used1)}")
    print(f"  Primo arrivo: {min(a[2] for a in used1):.7f} s  "
          f"(campione {int(round(min(a[2] for a in used1) * fs))})")

    print(f"\n⚙️  IR per RX2 — range {rr_rx2:.4f} m  [{label} arrivi per RD]")
    h2, used2 = build_ir(arr2, rd2_vals, rr_rx2, fs, n_arrivals=n_arr)
    print(f"  Spike totali nella IR: {len(used2)}")
    print(f"  Primo arrivo: {min(a[2] for a in used2):.7f} s  "
          f"(campione {int(round(min(a[2] for a in used2) * fs))})")

    if args.rx3:
        print(f"\n⚙️  IR per RX3 — range {rr_rx3:.4f} m  [{label} arrivi per RD]")
        h3, used3 = build_ir(arr3, rd3_vals, rr_rx3, fs, n_arrivals=n_arr)
        print(f"  Spike totali nella IR: {len(used3)}")
        print(f"  Primo arrivo: {min(a[2] for a in used3):.7f} s  "
              f"(campione {int(round(min(a[2] for a in used3) * fs))})")

    # ── Convoluzione ──────────────────────────────────────────────────
    print(f"\n🔁 Convoluzione...")
    rx1 = fftconvolve(source, h1)[:len(source)].astype(np.float32)
    rx2 = fftconvolve(source, h2)[:len(source)].astype(np.float32)

    signals = [rx1, rx2]
    if args.rx3:
        rx3 = fftconvolve(source, h3)[:len(source)].astype(np.float32)
        signals.append(rx3)

    gmax = max(np.max(np.abs(s)) for s in signals)
    rx1 = (rx1 / gmax).astype(np.float32)
    rx2 = (rx2 / gmax).astype(np.float32)

    # ── Salvataggio ───────────────────────────────────────────────────
    sf.write(args.out1, rx1, fs, subtype='PCM_16')
    sf.write(args.out2, rx2, fs, subtype='PCM_16')
    if args.rx3:
        rx3 = (rx3 / gmax).astype(np.float32)
        sf.write(args.out3, rx3, fs, subtype='PCM_16')

    # ── Verifica interna ──────────────────────────────────────────────
    from scipy.signal import correlate as xcorr

    def measure_lag(a, b, fs):
        corr = xcorr(b, a, mode='full')
        lag  = np.argmax(corr) - (len(a) - 1)
        return lag, lag / fs * 1e6

    print(f"\n{'═'*55}")
    print(f"  Arrivi per RD:      {label}")
    print(f"  {'Coppia':<12}  {'Δ range':>10}  {'Teorico':>12}  {'Misurato':>12}  {'Errore':>8}")
    print(f"  {'─'*53}")

    pairs = [('RX1→RX2', rr_rx1, rr_rx2, rx1, rx2)]
    if args.rx3:
        pairs.append(('RX1→RX3', rr_rx1, rr_rx3, rx1, rx3))
        pairs.append(('RX2→RX3', rr_rx2, rr_rx3, rx2, rx3))

    for name, ra, rb, sa, sb in pairs:
        lag, dt_meas = measure_lag(sa, sb, fs)
        dt_th = (rb - ra) / SOUND_SPEED * 1e6
        print(f"  {name:<12}  {rb-ra:>+10.4f}m  {dt_th:>+10.1f}µs  "
              f"{dt_meas:>+10.1f}µs  {abs(dt_meas-dt_th):>6.1f}µs")

    print(f"{'═'*55}")
    print(f"\n✅ Salvati:")
    print(f"   {os.path.abspath(args.out1)}")
    print(f"   {os.path.abspath(args.out2)}")
    if args.rx3:
        print(f"   {os.path.abspath(args.out3)}")


if __name__ == '__main__':
    main()
