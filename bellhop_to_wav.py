import numpy as np
from scipy.signal import fftconvolve, resample_poly
from math import gcd
import soundfile as sf

# ═══════════════════════════════════════════════════════════════════════
# PARAMETRI CONFIGURABILI
# ═══════════════════════════════════════════════════════════════════════

FS_OUT = 96000     # Hz — frequenza di campionamento output

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
        print("Nessun arrivo trovato per questo range!")
        return np.zeros(int(0.01 * fs), dtype=np.float32), []

    max_time = max(a[2] for a in used_arrivals)
    n = int(max_time * fs) + int(0.05 * fs)
    h = np.zeros(n, dtype=np.float64)

    for amp, phase, time in used_arrivals:
        sample = int(round(time * fs))
        if 0 <= sample < n:
            h[sample] += amp * np.cos(phase)

    return h.astype(np.float32), used_arrivals


# ===============================================================================================
def from_arr_to_wav(
    rx1: str,
    rx2: str,
    rx3: str,
    source: str,
    out1: str,
    out2: str,
    out3: str,
    fs: int = FS_OUT,
    n_arrivals: int = 1,
):
    """
    Genera file .wav simulati per RX1/RX2/RX3 da file .arr di Bellhop.

    Parametri
    ---------
    rx1        : percorso al file .arr per RX1
    rx2        : percorso al file .arr per RX2
    rx3        : percorso al file .arr per RX3
    source     : percorso al file audio sorgente
    out1/2/3   : percorsi di output per i .wav
    fs         : sample rate output
    dur        : durata sorgente sintetica in secondi
    fc         : frequenza centrale motore in Hz
    n_arrivals : arrivi per RD ordinati per tempo 
    """
    n_arr = n_arrivals
    label = str(n_arr) if n_arr > 0 else "tutti"

    src = load_audio_source(source, fs)

    # ── Lettura .arr ──────────────────────────────────────────────────
    print(f"\n📖 Lettura {rx1}...")
    rr1_vals, rd1_vals, arr1 = read_arr(rx1)
    rr_rx1 = max(rr1_vals)

    print(f"📖 Lettura {rx2}...")
    rr2_vals, rd2_vals, arr2 = read_arr(rx2)
    rr_rx2 = max(rr2_vals)

    print(f"📖 Lettura {rx3}...")
    rr3_vals, rd3_vals, arr3 = read_arr(rx3)
    rr_rx3 = max(rr3_vals)

    # ── Risposta impulsiva ─────────────────────────────────────────────
    h1, used1 = build_ir(arr1, rd1_vals, rr_rx1, fs, n_arrivals=n_arr)
    h2, used2 = build_ir(arr2, rd2_vals, rr_rx2, fs, n_arrivals=n_arr)
    h3, used3 = build_ir(arr3, rd3_vals, rr_rx3, fs, n_arrivals=n_arr)


    # ── Convoluzione ──────────────────────────────────────────────────
    transient = np.max([len(h1),len(h2),len(h3)]) - 1
    rx1_out = fftconvolve(src, h1)[transient:transient+len(src)].astype(np.float32)
    rx2_out = fftconvolve(src, h2)[transient:transient+len(src)].astype(np.float32)
    rx3_out = fftconvolve(src, h3)[transient:transient+len(src)].astype(np.float32)
        
    signals = [rx1_out, rx2_out, rx3_out]
    gmax = max(np.max(np.abs(s)) for s in signals)
    rx1_out = (rx1_out / gmax).astype(np.float32)
    rx2_out = (rx2_out / gmax).astype(np.float32)
    rx3_out = (rx3_out / gmax).astype(np.float32)

    # ── Salvataggio ───────────────────────────────────────────────────
    sf.write(out1, rx1_out, fs, subtype='PCM_16')
    sf.write(out2, rx2_out, fs, subtype='PCM_16')
    sf.write(out3, rx3_out, fs, subtype='PCM_16')