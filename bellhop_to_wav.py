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

    i += 1  # skip entire global header

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
        print(f"  Converted from stereo to mono")
    if fs_orig != fs_target:
        g = gcd(fs_target, fs_orig)
        up = fs_target // g
        down = fs_orig // g
        data = resample_poly(data, up, down).astype(np.float32)
        print(f"  Resampled from {fs_orig} Hz to {fs_target} Hz")
    else:
        print(f"  Sampling frequency: {fs_orig} Hz (no resampling)")
    data /= np.max(np.abs(data))
    print(f"  Duration: {len(data)/fs_target:.2f} s ({len(data)} samples)")
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
def from_arr_to_wav(
    input_folder: str,
    number_mic: int,
    source: str,
    out_folder: str,
    n_arrivals: int = 0,
    duration: int = 1,
):
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
    os.makedirs(out_folder, exist_ok=True)

    src = load_audio_source(source, FS_OUT)

    # ── Reading .arr ──────────────────────────────────────────────────
    arr_list  = []   # dictionaries with data from each microphone
    for i in range(1, number_mic + 1):
        arr_path = os.path.join(input_folder, f"H{i}.arr")
        #arr_path = f"{input_folder}/{i}.arr"
        #print(f"\nReading {arr_path}...")

        rr_vals, rd_vals, arr = read_arr(arr_path)
        arr_list.append({
            "rr_vals": rr_vals,
            "rd_vals": rd_vals,
            "arr":     arr,
            "rr_max":  max(rr_vals),
        })

    # ── Impulse response ────────────────────────────────────────────
    ir_list = []
    first_non_zero_at = []
    for i, mic in enumerate(arr_list, start=1):
        h, used = build_ir(mic["arr"], mic["rd_vals"], mic["rr_max"], FS_OUT, n_arrivals=n_arrivals)
        ir_list.append(h)
        first_non_zero_at.append(np.nonzero(h)[0][0])
        #print(f"  Hydrophone {i}: {len(used)} arrivals used, IR length = {len(h)} samples")
        #np.save("H.npy", h)

    # ── Convolution ──────────────────────────────────────────────────
    transient = max((len(h) for h in ir_list if h.size > 0), default=FS_OUT)
    maximum_length = max((len(h) for h in ir_list if h.size > 0))
    to_clip = np.max(first_non_zero_at)
    
    rx_out_list = []
    for h in ir_list:
        #h_bis = h[minimum_length-FS_OUT:minimum_length]
        rx_out = convolve(src[0:FS_OUT], h, mode='full', method='direct').astype(np.float32)
        #rx_out = convolve(src, h, mode='full', method='direct')[transient:transient + FS_OUT].astype(np.float32)
        #rx_out = convolve(src, h, mode='full', method='direct')[:FS_OUT].astype(np.float32)
        
        rx_out = rx_out[to_clip: to_clip + FS_OUT].astype(np.float32)
        rx_out_list.append(rx_out)

    # global normalization
    gmax = max(np.max(np.abs(s)) for s in rx_out_list)
    #gmax = max((np.max(np.abs(s)) for s in rx_out_list if s.size > 0), default=1.0)
    
    rx_out_list = [(s / gmax).astype(np.float32) for s in rx_out_list]

    # ── Saving ───────────────────────────────────────────────────
    out_paths = []
    for i, rx_out in enumerate(rx_out_list, start=1):
        out_path = os.path.join(out_folder, f"H{i}.npy")
        np.save(out_path, rx_out)