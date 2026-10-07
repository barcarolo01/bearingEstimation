import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from digitalshadow.devices.Floater import Floater
from digitalshadow.devices.IMU_models import load_imu_model
from digitalshadow.devices.Transmitter import Transmitter
from simulation import Simulation

SIMULATE = True

FONTSIZE = 14
FONTSIZE_LEGEND = 12

NUMBER_OF_FLOATERS = 10
N_SIMULATIONS = 50
STEPS = 3600
PERIODS = [60, 120, 180, 300, 600, 900, 1800]

# One entry per IMU case: results file and plot color
CASES = {
    "calibrated":   dict(file="MDS_calibrated.npz",   color='#2AB040', edge='#1d7a2c',
                         label="Calibrated IMU (no bias)"),
    "uncalibrated": dict(file="MDS_uncalibrated.npz", color='#E07B28', edge='#a3531a',
                         label="Uncalibrated IMU"),
}


def run_all(calibrated, results_file):
    Center = [32.839, -34.635]
    N_PERIODS = len(PERIODS)

    # One value (error averaged over floaters and time) per (period, run) pair
    err_imu = np.zeros((N_PERIODS, N_SIMULATIONS))
    err_mds = np.zeros((N_PERIODS, N_SIMULATIONS))

    for p, period in enumerate(PERIODS):
        print(f"[{'calibrated' if calibrated else 'uncalibrated'}] Period {period}")
        # Same generator for every period and for both cases
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
                f = load_imu_model(f, 'ADIS16470', dt=1.0, param_seed=9999+i)

                # Ideal calibration: remove accelerometer and gyroscope biases
                if calibrated:
                    f.accel_bias = np.zeros(3)
                    f.sigma_accel_bias = np.zeros(3)
                    f.gyro_bias = 0
                    f.sigma_gyro_bias = 0

                f.set_initial_velocity(rnd_sim.uniform(-1, 1), rnd_sim.uniform(-1, 1), 0.0)
                f.set_sigma(rnd_sim.uniform(0, 0.1), rnd_sim.uniform(0, 0.1), 0.0)
                f.use_compass = True
                f.alpha_compass = 0.999
                f.Rho = 0.999
                f.Rho_yaw = 0.999
                f.MDS_freq = period
                f.TX_LIMIT = 10
                floaters.append(f)

            sim = Simulation(Floaters=floaters,
                             Steps=STEPS,
                             Center=Center,
                             transmitter=TX,
                             seed=seed,
                             sim_name="sim_MDSFreq")
            sim.TX_LIMIT = 10
            sim.PACKET_LOSS = 0
            res = sim.run_simulation()

            err_imu[p, seed] = np.mean(res['errors_imu'])
            err_mds[p, seed] = np.mean(res['errors_imu_mds'])

    np.savez(results_file, periods=np.asarray(PERIODS), err_imu=err_imu, err_mds=err_mds)
    print(f"END OF SIMULATION -> {results_file}")
    return np.asarray(PERIODS), err_imu, err_mds


def plot_results(periods, results, outfile="MDS_calibration_comparison.png"):
    """results: dict case_name -> (err_imu, err_mds), same keys as CASES."""
    N_PERIODS = len(periods)
    x = np.arange(N_PERIODS)
    lo, hi = (25, 75)

    n_cases = len(results)
    BOX_WIDTH = 0.35
    offsets = (np.arange(n_cases) - (n_cases - 1) / 2) * (BOX_WIDTH + 0.05)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axhline(0, color='grey', linewidth=1)  # Grey line on y=0

    title_parts = []
    y_min = 0.0   # lowest plotted element (box bottom or mean), to avoid clipping negatives
    for (name, (err_imu, err_mds)), dx in zip(results.items(), offsets):
        style = CASES[name]
        n_runs = err_imu.shape[1]

        # Positive = MDS reduces the error
        improvement = (err_imu - err_mds) / err_imu * 100
        y_min = min(y_min, np.percentile(improvement, lo, axis=1).min(),
                    improvement.mean(axis=1).min())

        # Boxplot for IQR
        ax.boxplot(improvement.T, positions=x + dx, widths=BOX_WIDTH,
                   whis=(lo, hi), showcaps=False, showmeans=True, showfliers=False,
                   patch_artist=True,
                   boxprops=dict(facecolor=style['color'], alpha=0.35, edgecolor=style['edge']),
                   medianprops=dict(color=style['edge'], linewidth=2),
                   meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=6),
                   whiskerprops=dict(linewidth=0))

        # Median value label above each box
        medians = np.median(improvement, axis=1)
        box_tops = np.percentile(improvement, hi, axis=1)   # upper edge of each box
        for i in range(N_PERIODS):
            ax.annotate(f'{medians[i]:.1f}%',
                        (x[i] + dx, box_tops[i]),   # top edge of box
                        textcoords="offset points", xytext=(0, 3),
                        ha='center', va='bottom', fontsize=FONTSIZE_LEGEND - 2,
                        color=style['edge'])

        # IMU-only baseline (identical for every period): shown as a reference
        b_lo, b_med, b_hi = np.percentile(err_imu[0], [lo, 50, hi])
        title_parts.append(f"{name}: median {b_med:.0f} m (IQR {b_lo:.0f}–{b_hi:.0f} m)")

    ax.set_ylabel("Error reduction with MDS [%]", fontsize=FONTSIZE)
    ax.set_xlabel("MDS period [seconds]", fontsize=FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in periods])
    ax.margins(y=0.15)

    handles = [Patch(facecolor=CASES[n]['color'], alpha=0.35, edgecolor=CASES[n]['edge'],
                     label=CASES[n]['label']) for n in results]
    handles += [plt.Line2D([], [], marker='D', linestyle='', markerfacecolor='white',
                           markeredgecolor='black', label='mean'),
                plt.Line2D([], [], color='black', linewidth=2, label='median')]
    ax.legend(handles=handles, fontsize=FONTSIZE_LEGEND, loc='upper right',
              ncol=2, frameon=False, borderaxespad=0.2)

    ax.set_title("IMU-only error: " + ";  ".join(title_parts),
                 loc='left', fontsize=FONTSIZE_LEGEND)

    ax.tick_params(axis="both", which="both", labelsize=FONTSIZE)
    ax.grid(linestyle='--', alpha=0.5)
    # Start at 0 unless some box goes negative (MDS worse than IMU-only)
    ax.set_ylim(bottom=0 if y_min >= 0 else y_min - 0.1 * abs(ax.get_ylim()[1]))

    fig.tight_layout()
    fig.savefig(outfile, dpi=900)
    plt.show()


if __name__ == '__main__':
    results = {}
    for name, style in CASES.items():
        if SIMULATE:
            periods, err_imu, err_mds = run_all(calibrated=(name == "calibrated"),
                                                results_file=style['file'])
        else:
            d = np.load(style['file'])
            periods, err_imu, err_mds = d['periods'], d['err_imu'], d['err_mds']
        results[name] = (err_imu, err_mds)
    plot_results(periods, results)