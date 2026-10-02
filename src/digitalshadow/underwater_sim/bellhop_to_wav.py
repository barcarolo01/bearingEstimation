import os
from digitalshadow.utils.checkcal import check_calibration
from matplotlib import pyplot as plt
import numpy as np
from scipy.fft import rfft, irfft, rfftfreq, next_fast_len
from scipy.signal import resample_poly
from math import gcd
import soundfile as sf
from digitalshadow.underwater_sim.ambient_noise import get_noise_config, add_ambient_noise

FS_OUT = 96000
P_REF = 1e-6          # Pa  (0 dB re 1 µPa)

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


def active_rms(sig, fs, active_thresh_db=-40.0, frame_ms=20.0):
    """
    RMS computed only on 'active' frames, i.e. frames whose energy is within
    active_thresh_db of the loudest frame. Avoids silences biasing the level.
    """
    frame = max(1, int(fs * frame_ms / 1000))
    n_frames = len(sig) // frame
    if n_frames == 0:
        return np.sqrt(np.mean(sig ** 2))
    e = np.mean(sig[:n_frames * frame].reshape(n_frames, frame) ** 2, axis=1)
    active = e >= e.max() * 10 ** (active_thresh_db / 10)
    return np.sqrt(np.mean(e[active]))


def load_audio_source(filepath, fs_target, sl_db=None, active_thresh_db=-40.0):
    """
    Loads the source track. If sl_db is given, the track is scaled to physical
    pressure [Pa] at 1 m, such that its active RMS equals SL (dB re 1 µPa @ 1 m).
    """
    data, fs_orig = sf.read(filepath, dtype='float64')
    if data.ndim > 1:
        data = data.mean(axis=1)
        print("Converted from stereo to mono")

    if fs_orig != fs_target:
        g = gcd(fs_target, fs_orig)
        data = resample_poly(data, fs_target // g, fs_orig // g)
        print(f"Resampled from {fs_orig} Hz to {fs_target} Hz")

    data /= np.max(np.abs(data))

    if sl_db is not None:
        p_rms_target = P_REF * 10 ** (sl_db / 20)          # Pa @ 1 m
        data *= p_rms_target / active_rms(data, fs_target, active_thresh_db)
        print(f"Source calibrated: SL = {sl_db:.1f} dB re 1 µPa @ 1 m "
              f"(RMS = {p_rms_target:.3g} Pa)")
    return data


def select_arrivals(arrivals_dict, rd_values, rr_target, n_arrivals=0):
    """Returns the list of (amp, phase_deg, time_s) used for one hydrophone."""
    used = []
    for rd in rd_values:
        arr_list = arrivals_dict.get((round(rd, 6), round(rr_target, 3)), [])
        sorted_arr = sorted(arr_list, key=lambda a: a[2])
        used.extend(sorted_arr[:n_arrivals] if n_arrivals > 0 else sorted_arr)
    return used


def synth_rx(src, arrivals, fs, n_out, t0, phase_sign=+1, chunk=16):
    """
    Received pressure in the window [t0, t0 + n_out/fs), computed in the
    frequency domain with full complex amplitude and exact (fractional) delays:

        Y(f) = X(f) * sum_k A_k * exp(i*phase_sign*phi_k) * exp(-i*2*pi*f*(tau_k - t0)),  f >= 0

    Equivalent to  y(t) = sum_k A_k [cos(phi) s(t-tau) - sin(phi) Hilbert{s}(t-tau)]
    (same convention as delayandsum.m in the Acoustics Toolbox).
    """
    arr = np.asarray(arrivals, dtype=np.float64).reshape(-1, 3)

    # Amplitude, phase and time of travel
    amp, ph, tau = arr[:, 0], arr[:, 1], arr[:, 2]
    d = tau - t0

    # keep only arrivals that contributes to the OUT file
    keep = (d >= 0) & (d < n_out / fs)
    amp, ph, d = amp[keep], ph[keep], d[keep]

    # Input signal clipping
    x = np.zeros(n_out)
    x[:min(n_out, len(src))] = src[:n_out]

    nfft = next_fast_len(2 * n_out) 
    X = rfft(x, n=nfft)
    f = rfftfreq(nfft, 1 / fs)

    c = amp * np.exp(1j * phase_sign * np.deg2rad(ph))
    H = np.zeros(len(f), dtype=np.complex128)
    for k in range(0, len(c), chunk):
        H += (c[k:k + chunk, None] * np.exp(-2j * np.pi * np.outer(d[k:k + chunk], f))).sum(axis=0)

    return irfft(X * H, n=nfft)[:n_out]


# ===============================================================================================
def from_arr_to_wav(input_folder: str, number_mic: int, source: str, out_folder: str,
                    n_arrivals=0, sl_db=150, phase_sign=+1,
                    add_noise=False, noise_seed=1, save_clean=False):
    """
    Generates the received pressure [Pa] for N hydrophones from Bellhop .arr files.

    Parameters
    ----------
    input_folder : folder containing the .arr files (H1.arr, H2.arr, ...)
    number_mic   : number of .arr files to read
    source       : path to the source audio file
    out_folder   : folder where output .npy files will be saved
    n_arrivals   : number of arrivals per RD sorted by time (0 = all)
    sl_db        : source level, dB re 1 µPa @ 1 m (None = old arbitrary scale)
    phase_sign   : sign convention for Bellhop phases (+1 as in delayandsum.m)
    add_noise    : True/False forza il rumore ambientale; None = usa il flag
                   globale ADD_AMBIENT_NOISE (variabile d'ambiente)
    noise_seed   : seed del rumore; None = usa NOISE_SEED (se definito) o casuale
    save_clean   : se True salva anche le tracce senza rumore (H{i}_clean.npy)
    """
    os.makedirs(out_folder, exist_ok=True)
    src = load_audio_source(source, FS_OUT, sl_db=sl_db)

    # == Arrivals for each hydrophone
    arrivals_per_mic = []
    for i in range(1, number_mic + 1):
        rr_vals, rd_vals, arr = read_arr(os.path.join(input_folder, f"H{i}.arr"))
        used = select_arrivals(arr, rd_vals, max(rr_vals), n_arrivals)
        if not used:
            raise ValueError(f"Hydrophone {i}: no arrivals found")
        arrivals_per_mic.append(used)

    # == Common time reference: earliest arrival over all hydrophones
    t0 = min(min(a[2] for a in used) for used in arrivals_per_mic)

    # == Synthesis + calibration check (in Pa)
    n_out = FS_OUT
    clean = []
    for i, used in enumerate(arrivals_per_mic, start=1):
        rx = synth_rx(src, used, FS_OUT, n_out, t0, phase_sign=phase_sign)
        check_calibration(rx, src, used, FS_OUT, sl_db, label=f"H{i}")
        clean.append(rx)

    # == Ambient noise (Wenz), optional
    cfg = get_noise_config()

    if add_noise:
        rng = np.random.default_rng(noise_seed)
        print(f"Ambient noise ON: shipping={cfg['shipping']}, wind={cfg['wind']} m/s, "
              f"extra={cfg['extra_db']:+.1f} dB, band {cfg['f_low']:.0f}-{cfg['f_high']:.0f} Hz, "
              f"seed={noise_seed}")
        out, _ = add_ambient_noise(clean, FS_OUT,
                                   shipping=cfg["shipping"], wind=cfg["wind"],
                                   extra_db=cfg["extra_db"], rng=rng,
                                   f_low=cfg["f_low"], f_high=cfg["f_high"])
    else:
        print("Ambient noise OFF")
        out = clean

    # == Save
    for i, rx in enumerate(out, start=1):
        np.save(os.path.join(out_folder, f"H{i}.npy"), rx)
        if save_clean and add_noise:
            np.save(os.path.join(out_folder, f"H{i}_clean.npy"), clean[i - 1])



# ===============================================================================================
def build_ir(arrivals_dict, rd_values, rr_target, fs, n_arrivals=1):
    """Kept only for plotting (plot_ir). Not used for the synthesis anymore."""
    used_arrivals = select_arrivals(arrivals_dict, rd_values, rr_target, n_arrivals)
    if not used_arrivals:
        print("No arrivals found for this range!")
        return np.zeros(int(0.01 * fs), dtype=np.float32), []

    max_time = max(a[2] for a in used_arrivals)
    n = int(max_time * fs) + int(0.05 * fs)
    h = np.zeros(n, dtype=np.float64)
    for amp, phase, time in used_arrivals:
        sample = int(round(time * fs))
        if 0 <= sample < n:
            h[sample] += amp * np.cos(np.deg2rad(phase))
    return h.astype(np.float32), used_arrivals


def plot_ir(used_arrivals, fs, out_path=None, db_panel=True, title=None):
    """
    Plots the channel impulse response as a stem plot.
    """
    if not used_arrivals:
        raise ValueError("Empty arrival list: nothing to plot")

    amp = np.array([a[0] for a in used_arrivals])
    phase = np.array([a[1] for a in used_arrivals])
    time = np.array([a[2] for a in used_arrivals])

    t_rel = (time - time.min()) * 1e3
    a_signed = amp * np.cos(np.deg2rad(phase))

    nrows = 2 if db_panel else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(7, 5.0 if db_panel else 3.2), sharex=True)
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ax.stem(t_rel, a_signed, basefmt=" ", markerfmt="o", linefmt="-")
    ax.axhline(0, lw=0.6, color="k")
    ax.set_ylabel("Amplitude", fontsize=18)
    ax.grid(alpha=0.3)

    if db_panel:
        axes[1].stem(t_rel, 20 * np.log10(amp), basefmt=" ")
        axes[1].set_ylabel("|A| [dB]", fontsize=18)
        axes[1].grid(alpha=0.3)

    axes[-1].set_xlabel("Delay relative to the first arrival [ms]", fontsize=18)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    return fig