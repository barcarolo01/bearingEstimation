import os

import numpy as np
from scipy.signal import convolve, resample_poly
from math import gcd
import soundfile as sf

FS_OUT = 96000     # Sampling frequency of output files (Hz)

def read_arr(filename):
    with open(filename) as f:
        lines = [l.strip() for l in f.readlines()]

    i = 2 # Skipping header rows ('2D' and frequency value)
    
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
    input_folder: str,
    number_mic: int,
    source: str,
    out_folder: str,
    n_arrivals: int = 0,
    duration: int = 1,
):
    """
    Genera file .wav simulati per N microfoni da file .arr di Bellhop.

    Parametri
    ---------
    input_folder : cartella contenente i file .arr (rx1.arr, rx2.arr, ...)
    number_mic   : numero di file .arr da leggere
    source       : percorso al file audio sorgente
    out_folder   : cartella dove salvare i file .wav di output
    n_arrivals   : numero di arrivi per RD ordinati per tempo (0 = tutti)
    fs           : sample rate output
    """
    os.makedirs(out_folder, exist_ok=True)

    src = load_audio_source(source, FS_OUT)

    # ── Lettura .arr ──────────────────────────────────────────────────
    arr_list  = []   # dizionari con i dati di ogni microfono
    for i in range(1, number_mic + 1):
        arr_path = os.path.join(input_folder, f"H{i}.arr")
        #arr_path = f"{input_folder}/{i}.arr"
        #print(f"\nLettura {arr_path}...")

        rr_vals, rd_vals, arr = read_arr(arr_path)
        arr_list.append({
            "rr_vals": rr_vals,
            "rd_vals": rd_vals,
            "arr":     arr,
            "rr_max":  max(rr_vals),
        })

    # ── Risposta impulsiva ────────────────────────────────────────────
    ir_list = []
    for i, mic in enumerate(arr_list, start=1):
        h, used = build_ir(mic["arr"], mic["rd_vals"], mic["rr_max"],FS_OUT, n_arrivals=n_arrivals)
        ir_list.append(h)

    # ── Convoluzione ──────────────────────────────────────────────────
    transient = max(len(h) for h in ir_list) - 1

    rx_out_list = []
    for h in ir_list:
        rx_out = convolve(src, h, mode='full', method='direct')[transient:transient + FS_OUT].astype(np.float32)
        #rx_out = convolve(src, h, mode='full', method='direct')[:FS_OUT].astype(np.float32)
        rx_out_list.append(rx_out)

    # Normalizzazione globale
    gmax = max(np.max(np.abs(s)) for s in rx_out_list)
    rx_out_list = [(s / gmax).astype(np.float32) for s in rx_out_list]

    # ── Salvataggio ───────────────────────────────────────────────────
    out_paths = []
    for i, rx_out in enumerate(rx_out_list, start=1):
        out_path = os.path.join(out_folder, f"H{i}.npy")
        np.save(out_path,rx_out)