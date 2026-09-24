from digitalshadow.devices.IMU_models import load_imu_model
import numpy as np
import matplotlib.pyplot as plt
from digitalshadow.devices.Floater import *

plt.style.use('seaborn-v0_8-deep')

NUMBER_OF_RUNS = 20
STEPS = 3600

FONTSIZE = 14
FONTSIZE_LEGEND = 12

ALPHAS = [0.0, 0.5, 0.9, 0.98, 0.995, 0.999]

USE_CALIBRATED_GYROSCOPE = True

def build_floater(seed, alpha=None):
    """Still floater; alpha = None means gyroscope only (no compass)."""
    np.random.seed(seed)
    f = Floater(0, 0, 0, 0, 1, 1.0, imu_seed=seed)
    f = load_imu_model(f,'ADIS16470',dt=1.0)
    
    f.set_initial_velocity(0.0, 0.0, 0.0)
    f.Rho = 1
    f.set_sigma(0.0, 0.0, 0.0)
    f.sigma_yaw_rate = 0.0

    if USE_CALIBRATED_GYROSCOPE:
        f.sigma_gyro_bias = 0
        f.gyro_bias = 0

    if alpha is None:
        f.use_compass = False
    else:
        f.use_compass = True
        f.alpha_compass = alpha

    return f


def run_MC_simulations(alpha, NUMBER_OF_RUNS=NUMBER_OF_RUNS, steps=STEPS):
    err = np.zeros((NUMBER_OF_RUNS, steps))
    for r in range(NUMBER_OF_RUNS):
        f = build_floater(r, alpha)
        for k in range(steps):
            f.move()
            err[r, k] = f.psi_err
    return np.sqrt(np.mean(err**2, axis=0)), err




if __name__ == "__main__":
    t = np.arange(1, STEPS + 1)

    rms_gyro, _ = run_MC_simulations(None)
    plt.loglog(t, np.rad2deg(rms_gyro), lw=2.2, color="C3",label="gyroscope only")

    finals = []
    for a in ALPHAS:
        rms, _ = run_MC_simulations(a)
        plt.loglog(t, np.rad2deg(rms), lw=1.4, label=rf"$\alpha$ = {a}")
        finals.append(np.rad2deg(rms[-1]))

    plt.axhline(np.rad2deg(COMPASS_BIAS), color="0.4", ls=":", lw=1.2)
    plt.text(t[-1], np.rad2deg(COMPASS_BIAS), r"  $\beta$", color="0.4",
            fontsize=9, va="center")

    plt.xlabel("Time [seconds]", fontsize=FONTSIZE)
    plt.ylabel("Heading error [degrees, unwrapped]",fontsize=FONTSIZE)
    plt.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    plt.tick_params(axis="both", which="minor", labelsize=FONTSIZE)
    #plt.title("Heading error")
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.legend(fontsize=FONTSIZE_LEGEND)
    plt.savefig("compass_alpha_sweep.png",dpi=900)
    plt.show()