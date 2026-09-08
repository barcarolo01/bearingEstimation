import os
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
from scipy.signal import resample_poly
from math import gcd
import soundfile as sf

FS_OUT = 96000

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

    i += 1  # Skip entire global header

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
        print(f"Converted from stereo to mono")

    # Resample the track if its current sample rate does not match the target one
    if fs_orig != fs_target:
        g = gcd(fs_target, fs_orig)
        up = fs_target // g
        down = fs_orig // g
        data = resample_poly(data, up, down).astype(np.float32)
        print(f"Resampled from {fs_orig} Hz to {fs_target} Hz")

    data /= np.max(np.abs(data)) # Normalization
    return data

def build_ir(arrivals_dict, rd_values, rr_target, fs, n_arrivals=1):
    """
    Builds the impulse response from Bellhop arrivals.

    For each RD, selects the first N arrivals in chronological order
    and sums them into the IR as spikes (amplitude * cos(phase)) at their
    corresponding sample.

    Parameters
    ----------
    arrivals_dict : dict {(rd, rr): [(amp, phase, time), ...]}
    rd_values     : list of receiver depths
    rr_target     : receiver range of interest (m)
    fs            : sampling frequency (Hz)
    n_arrivals    : number of arrivals to use per RD, sorted by time.
                    0 = use all available arrivals.

    Returns
    -------
    h             : impulse response (float32)
    used_arrivals : total list of (amp, phase, time) inserted into the IR
    """
    used_arrivals = []

    for rd in rd_values:
        key = (round(rd, 6), round(rr_target, 3))
        arr_list = arrivals_dict.get(key, [])
        if not arr_list:
            continue

        # Sort by ascending time
        sorted_arr = sorted(arr_list, key=lambda a: a[2])

        # Select the first N (0 = all)
        if n_arrivals > 0:
            selected = sorted_arr[:n_arrivals]
        else:
            selected = sorted_arr

        used_arrivals.extend(selected)

    if not used_arrivals:
        print("No arrivals found for this range!")
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
def from_arr_to_wav(input_folder: str,number_mic: int,source: str,out_folder: str,n_arrivals=0):
    """
    Generates simulated .wav files for N microphones from Bellhop .arr files.

    Parameters
    ----------
    input_folder : folder containing the .arr files (rx1.arr, rx2.arr, ...)
    number_mic   : number of .arr files to read
    source       : path to the source audio file
    out_folder   : folder where output .wav files will be saved
    n_arrivals   : number of arrivals per RD sorted by time (0 = all)
    fs           : output sample rate
    """

    # Create the output folder if it does not exists
    os.makedirs(out_folder, exist_ok=True)

    # Load the audio source (with resample, if needed)
    src = load_audio_source(source, FS_OUT)

    # == Reading .arr file
    arr_list  = []   # dictionaries with data from each microphone
    for i in range(1, number_mic + 1):
        arr_path = os.path.join(input_folder, f"H{i}.arr")

        rr_vals, rd_vals, arr = read_arr(arr_path)
        arr_list.append({
            "rr_vals": rr_vals,
            "rd_vals": rd_vals,
            "arr":     arr,
            "rr_max":  max(rr_vals),
        })

    # == Impulse response
    ir_list = []
    first_non_zero_at = []
    for i, mic in enumerate(arr_list, start=1):
        # Create the impulse responde
        h, used = build_ir(mic["arr"], mic["rd_vals"], mic["rr_max"], FS_OUT, n_arrivals=n_arrivals)
        ir_list.append(h)

        nz = np.nonzero(h)[0]
        if nz.size == 0:
            raise ValueError(f"Hydrophone {i}: impulse response is null")

        # Keep track of the index of the first non-zero sample
        first_non_zero_at.append(nz[0])

    # Synch point: first time instant in which any track is not zero
    start = int(np.min(first_non_zero_at))

    # ── Sorgente: solo i campioni che possono influenzare l'uscita ──
    Lx = FS_OUT + start
    if len(src) < Lx:
        x = np.zeros(Lx, dtype=np.float32)
        x[:len(src)] = src
    else:
        x = src[:Lx].astype(np.float32)

    # ── Bound FISSO, indipendente dalla lunghezza reale delle IR ──
    max_hslice_len = Lx + FS_OUT - 1                 # limite superiore teorico del ritaglio di h
    nfft = next_fast_len(Lx + max_hslice_len - 1)    # stessa FFT size per TUTTI i canali
    X_f = rfft(x, n=nfft)                            # FFT della sorgente: calcolata UNA volta, riusata sempre

    rx_out_list = []
    for h in ir_list:
        k_lo = max(0, start - Lx + 1)
        k_hi = min(len(h) - 1, start + FS_OUT - 1)

        if k_hi < k_lo:
            # nessuna sovrapposizione possibile: canale silente nella finestra richiesta
            rx_out_list.append(np.zeros(FS_OUT, dtype=np.float32))
            continue

        h_slice = h[k_lo:k_hi + 1]                   # <-- indipendente da len(h) reale, bounded
        H_f = rfft(h_slice, n=nfft)
        z = irfft(X_f * H_f, n=nfft)                 # z[m] == y[m + k_lo]

        lo = start - k_lo                             # offset dentro z corrispondente a n = start
        rx_out = z[lo: lo + FS_OUT].astype(np.float32)

        if rx_out.size < FS_OUT:                      # safety net, non dovrebbe mai scattare
            rx_out = np.pad(rx_out, (0, FS_OUT - rx_out.size))

        rx_out_list.append(rx_out)

    # Global normalization
    #gmax = max(np.max(np.abs(s)) for s in rx_out_list)
    #rx_out_list = [(s / gmax).astype(np.float32) for s in rx_out_list]

    # Save the convolved track (one per hydrophone) in numpy array format (.npy)
    for i, rx_out in enumerate(rx_out_list, start=1):
        out_path = os.path.join(out_folder, f"H{i}.npy")
        np.save(out_path, rx_out)