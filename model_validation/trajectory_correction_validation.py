"""
Batch di N simulazioni + grafico dell'errore di localizzazione nel tempo.

Per ogni simulazione e per ogni floater calcola, a ogni timestamp, la distanza
euclidea tra la posizione stimata e quella reale per:
    - RX_imu       (blu)
    - RX_imu_mds   (verde)
    - RX_imu_comp  (ocra)
Poi aggrega su tutti i floater di tutte le simulazioni: media + banda IQR (25-75%).
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from digitalshadow.devices.Floater import *
from digitalshadow.devices.IMU_models import load_imu_model, calibrate_floater
from digitalshadow.devices.Transmitter import *
from simulation import *

# --------------------------------------------------------------------------- #
# Configurazione
# --------------------------------------------------------------------------- #
STEPS=3600
N_SIM = 3                 # numero di simulazioni del batch
BASE_SEED = 100            # seed della simulazione k = BASE_SEED + k
N_JOBS = 1            
NUMBER_OF_FLOATERS = 10
HYDROMATE_SIMULATION = False
TX_LIMIT = 3
DT = 1.0

FONTSIZE = 14
FONTSIZE_LEGEND = FONTSIZE -2

GT_KEY = "RX_gt_pos"        # riferimento ground truth (vedi nota nel messaggio)
ERROR_DIMS = 2              # 2 = errore orizzontale (x,y), 3 = errore 3D
OUT_DIR = "batch_results"
RESULTS_FILE = os.path.join(OUT_DIR, "errors.npz")

CENTER = [45.92458622396349, -20.45658040956359]

ESTIMATES = {
    # chiave nel dict restituito : (etichetta, colore)
    "RX_imu":      ("IMU",            "#0000FF"),
    "RX_imu_mds":  ("IMU + MDS",      "#2AB040"),
    "RX_imu_comp": ("IMU compensated", "#D6940F"),
}





def build_floaters(seed):
    rng = np.random.default_rng(seed)
    floaters = []
    for i in range(NUMBER_OF_FLOATERS):
        x = rng.uniform(-500,500)
        y = rng.uniform(-500,500)
        f = Floater(ID=i, gt_x=x, gt_y=y, gt_z=10,
                    NF=NUMBER_OF_FLOATERS, dt=DT,
                    seed=seed * 100 + i + 1)
        f = load_imu_model(f, 'ADIS16470', dt=DT)
        f.set_initial_velocity(rng.uniform(-2,2), rng.uniform(-2,2), 0.0)
        f = calibrate_floater(f)
        f.set_sigma(rng.uniform(0.01), rng.uniform(0.01), 0.0) 
        f.Rho = 0.999
        f.use_compass = True
        f.MDS_freq = 300
        f.TX_LIMIT = TX_LIMIT
        floaters.append(f)
    return floaters


def position_error(est, gt):
    """est, gt: [M, F, 3] -> errore euclideo [M, F] (troncato alla lunghezza comune)."""
    est, gt = np.asarray(est, dtype=float), np.asarray(gt, dtype=float)
    m = min(est.shape[0], gt.shape[0])
    diff = est[:m, :, :ERROR_DIMS] - gt[:m, :, :ERROR_DIMS]
    return np.linalg.norm(diff, axis=-1)


def run_one(run_idx):
    seed = BASE_SEED + run_idx
    TX = Transmitter(-1,0,0,10,1)
    sim = Simulation(transmitter=TX,
                     Floaters=build_floaters(seed),
                     Center=CENTER,
                     seed=seed,
                     Steps=STEPS,
                     sim_name=f"batch_{run_idx:03d}",
                     HYDROMATE_SIMULATION=HYDROMATE_SIMULATION)
    sim.PACKET_LOSS = 0.0
    res = sim.run_simulation()

    gt = res[GT_KEY]
    errs = {k: position_error(res[k], gt) for k in ESTIMATES}
    print(f"[run {run_idx:03d}] seed={seed} completata")
    return errs


# --------------------------------------------------------------------------- #
# Batch, salvataggio e aggregazione
# --------------------------------------------------------------------------- #
def run_batch():
    os.makedirs(OUT_DIR, exist_ok=True)
    if N_JOBS > 1:
        with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
            runs = list(ex.map(run_one, range(N_SIM)))
    else:
        runs = [run_one(k) for k in range(N_SIM)]

    # Le simulazioni possono avere lunghezze diverse: tronco alla minima
    m_min = min(r[k].shape[0] for r in runs for k in ESTIMATES)
    stacked = {k: np.stack([r[k][:m_min] for r in runs]) for k in ESTIMATES}  # [N, M, F]
    np.savez(RESULTS_FILE, **stacked)
    print(f"Errori salvati in {RESULTS_FILE}")
    return stacked


def aggregate(err):
    """err: [N, M, F] -> media, q25, q75 su tutti i floater di tutte le simulazioni."""
    n, m, f = err.shape
    pooled = err.transpose(1, 0, 2).reshape(m, n * f)   # [M, N*F]
    mean = np.nanmean(pooled, axis=1)
    q25, q75 = np.nanpercentile(pooled, [25, 75], axis=1)
    return mean, q25, q75


def plot(stacked):
    fig, ax = plt.subplots(figsize=(8, 6))
    n_sim, n_steps, n_flo = next(iter(stacked.values())).shape
    t = np.arange(n_steps) * DT

    for key, (label, color) in ESTIMATES.items():
        mean, q25, q75 = aggregate(stacked[key])
        print(f"{label}: errore medio = {np.nanmean(mean):.2f} m")
        ax.plot(t, mean, color=color, lw=2, label=f"{label}")
        ax.fill_between(t, q25, q75, color=color, alpha=0.2, lw=0)

    dims = "orizzontale (x-y)" if ERROR_DIMS == 2 else "3D"
    ax.set_xlabel("Simulation steps [seconds]",fontsize=FONTSIZE)
    ax.set_ylabel(f"Floater self-localization error [meters]",fontsize=FONTSIZE)
    #ax.set_title(f"Errore di localizzazione — {n_sim} simulazioni × {n_flo} floater")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=FONTSIZE_LEGEND)
    ax.tick_params('both',labelsize=FONTSIZE)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "error_vs_time.png")
    fig.savefig(out, dpi=900)
    print(f"Grafico salvato in {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="non rilancia le simulazioni, usa errors.npz salvato")
    args = parser.parse_args()

    if args.plot_only:
        data = np.load(RESULTS_FILE)
        stacked = {k: data[k] for k in ESTIMATES}
    else:
        stacked = run_batch()
    plot(stacked)