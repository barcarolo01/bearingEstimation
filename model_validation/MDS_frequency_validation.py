import numpy as np
import matplotlib.pyplot as plt
from digitalshadow.devices.Floater import Floater
from digitalshadow.devices.IMU_models import load_imu_model
from digitalshadow.devices.Transmitter import Transmitter
from simulation import Simulation

SIMULATE = False
RESULTS_FILE = "MDS_results.npz"

FONTSIZE = 14
FONTSIZE_LEGEND = 12

NUMBER_OF_FLOATERS = 4
N_SIMULATIONS = 10
STEPS = 3600
PERIODS = [60, 120, 180, 300, 600, 900, 1200, 1800]

def run_all():
    Center = [32.839, -34.635]
    N_PERIODS = len(PERIODS)

    # One value (error averaged over floaters and time) per (period, run) pair
    err_imu = np.zeros((N_PERIODS, N_SIMULATIONS))
    err_mds = np.zeros((N_PERIODS, N_SIMULATIONS))

    for p, period in enumerate(PERIODS):
        # Same generator for every period
        rnd_sim = np.random.default_rng(256123)

        for seed in range(N_SIMULATIONS):
            TX = Transmitter(-1, 0, 0, 0, 1.0)
            TX.set_initial_velocity(1, 1, 0)
            TX.set_sigma(0.1, 0.1, 0)
            TX.Rho = 0.999

            floaters = []
            for i in range(NUMBER_OF_FLOATERS):
                f = Floater(ID=i,
                            gt_x=rnd_sim.uniform(-500, 500),
                            gt_y=rnd_sim.uniform(-500, 500),
                            gt_z=10,
                            NF=NUMBER_OF_FLOATERS,
                            dt=1.0)
                f = load_imu_model(f, 'ADIS16470', dt=1.0, param_seed=9999)
                f.set_initial_velocity(rnd_sim.uniform(-1, 1), rnd_sim.uniform(-1, 1), 0.0)
                f.set_sigma(rnd_sim.uniform(0, 0.1), rnd_sim.uniform(0, 0.1), 0.0)
                f.use_compass = False
                f.Rho = 0.999
                f.Rho_yaw = 0.999
                f.MDS_freq = period
                f.TX_LIMIT = 10
                floaters.append(f)

            sim = Simulation(Floaters=floaters,
                             Steps=STEPS,
                             Center=Center,
                             transmitter=TX,
                             seed=seed)
            sim.TX_LIMIT = 10
            sim.PACKET_LOSS = 0
            res = sim.run_simulation()

            err_imu[p, seed] = np.mean(res['errors_imu'])
            err_mds[p, seed] = np.mean(res['errors_imu_mds'])


    np.savez(RESULTS_FILE, periods=np.asarray(PERIODS), err_imu=err_imu, err_mds=err_mds)
    return np.asarray(PERIODS), err_imu, err_mds


def plot_results(periods, err_imu, err_mds, outfile="MDS_example.png"):
    N_PERIODS, n_runs = err_imu.shape
    x = np.arange(N_PERIODS) 
    lo, hi = (25,75)

    # Positive = MDS reduces the error
    improvement = (err_imu - err_mds) / err_imu * 100

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.axhline(0, color='grey', linewidth=1) # Grey line on y=0

    # Boxplot for IQR
    ax.boxplot(improvement.T, positions=x, widths=0.5,
               whis=(lo, hi), showcaps=False, showmeans=False, showfliers=False,
               patch_artist=True,boxprops=dict(facecolor='#2AB040', alpha=0.35, edgecolor='#1d7a2c'),
               medianprops=dict(color='#1d7a2c', linewidth=2),
               meanprops=dict(marker='D', markerfacecolor='white',markeredgecolor='black', markersize=6),
               whiskerprops=dict(linewidth=0))
    
    # Median value label above each box
    medians = np.median(improvement, axis=1)
    BOX_WIDTH = 0.5   # same value passed to boxplot(widths=...)
    for i in range(N_PERIODS):
        ax.annotate(f'{medians[i]:.1f}%',
                    (x[i] + BOX_WIDTH / 2, medians[i]),   # right edge of box, at the median
                    textcoords="offset points", xytext=(4, 0),
                    ha='left', va='center', fontsize=FONTSIZE_LEGEND - 1,
                    color='#1d7a2c')

    ax.set_ylabel("Error reduction with MDS [%]", fontsize=FONTSIZE)
    ax.set_xlabel("MDS period [seconds]", fontsize=FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in periods])
    ax.margins(y=0.15)
    ax.plot([], [], 'D', markerfacecolor='white', markeredgecolor='black', label='mean')
    ax.plot([], [], '-', color='#1d7a2c', linewidth=2, label='median')
    

    ax.legend(fontsize=FONTSIZE_LEGEND, loc='upper right',ncol=2, frameon=False, borderaxespad=0.2,
              title_fontsize=FONTSIZE_LEGEND)

    # IMU-only baseline (identical for every period): shown as a reference
    imu_run = err_imu[0]
    b_lo, b_med, b_hi = np.percentile(imu_run, [lo, 50, hi])
    ax.set_title(f"IMU-only error: median {b_med:.0f} m  "
                 f"(IQR {b_lo:.0f}–{b_hi:.0f} m)",loc='left', fontsize=FONTSIZE_LEGEND)

    ax.tick_params(axis="both", which="both", labelsize=FONTSIZE)
    ax.grid(linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(outfile, dpi=900)
    plt.show()


if __name__ == '__main__':
    if SIMULATE:
        periods, err_imu, err_mds = run_all()
    else:
        d = np.load(RESULTS_FILE)
        periods, err_imu, err_mds = d['periods'], d['err_imu'], d['err_mds']
    plot_results(periods, err_imu, err_mds)