import numpy as np
import matplotlib.pyplot as plt
from digitalshadow.devices.Floater import *
from digitalshadow.devices.IMU_models import *

plt.style.use('seaborn-v0_8-deep')

NUMBER_OF_RUNS = 20
STEPS = 3600

FONTSIZE = 14
FONTSIZE_LEGEND = 12

SCENARIOS = {
    "white noise only":  dict(bias=False, white=True,  rw=False),
    "random walk only":  dict(bias=False, white=False, rw=True),
    "constant bias only": dict(bias=True,  white=False, rw=False),
    "all sources":       dict(bias=True,  white=True,  rw=True),
}

# Expected asymptotic slope in log-log, for reference lines
EXPECTED_SLOPE = {
    "white noise only": 0.5,
    "random walk only": 1.5,
    "constant bias only": 1.0,
}


def build_floater(seed, cfg):
    """Create a static floater with only the selected gyroscope error sources enabled."""
    np.random.seed(seed)
    f = Floater(0, 0, 0, 0, 1, 1.0, imu_seed=seed)

    # Static: the heading error does not depend on the motion, but keeping the
    # floater still isolates the rotational part completely
    f.set_initial_velocity(0.0, 0.0, 0.0)
    f.Rho = 1
    f.set_sigma(0.0, 0.0, 0.0)
    f.sigma_yaw_rate = 0.0

    f = load_imu_model(f,'ADIS16470',dt=1.0)
    # Simulator side: silence the sources that are off
    if not cfg["bias"]:
        f.gyro_bias = 0
        f.sigma_gyro_bias = 0
    if not cfg["white"]:
        f.sigma_gyro_white_noise = 0.0
    if not cfg["rw"]:
        f.sigma_gyro_bias_driving = 0.0

    # Estimator side: the analytical model must see the same configuration
    if not cfg["bias"]:
        f.sigma_gyro_bias = 0.0

    f.accel_bias = np.zeros(3)
    f.sigma_accel_bias = np.zeros(3)
    f.sigma_accel_white_noise = np.zeros(3)
    f.sigma_accel_bias_driving = np.zeros(3)
    f.sigma = np.zeros(3)

    return f


def heading_model(f, N):
    """Closed form of the heading uncertainty after N steps."""
    dt = f.dt
    sum2 = N * (N + 1) * (2 * N + 1) / 6.0

    sigma_psi_bias = f.sigma_gyro_bias * dt * N
    sigma_psi_white = f.sigma_gyro_white_noise * dt * np.sqrt(N)
    sigma_psi_rw = f.sigma_gyro_bias_driving * dt * np.sqrt(sum2)

    return np.sqrt(sigma_psi_bias**2 + sigma_psi_white**2 + sigma_psi_rw**2)


def run_MC_simulations(cfg, NUMBER_OF_RUNS=NUMBER_OF_RUNS, steps=STEPS):
    err = np.zeros((NUMBER_OF_RUNS, steps))
    pred = np.zeros(steps)  # Heading error predicted by the analytical model

    for r in range(NUMBER_OF_RUNS):
        f = build_floater(r, cfg)
        for k in range(steps):
            f.move()
            # Unwrapped heading error: the wrapped one saturates at 180 deg
            err[r, k] = f.psi_err_unwrapped

            # Analytical model is deterministic: we compute it only at the first run
            if r == 0:
                pred[k] = heading_model(f, k + 1)

    # The heading error is a scalar, so the model directly predicts its RMS
    return np.sqrt(np.mean(err**2, axis=0)), pred, err


if __name__ == "__main__":
    t = np.arange(1, STEPS + 1)

    # FIGURE 1
    for name, slope in EXPECTED_SLOPE.items():
        rms, pred, _ = run_MC_simulations(SCENARIOS[name])
        plt.loglog(t, np.rad2deg(rms), lw=1.8, label=f"{name}")

        dark_col = "#343434"
        plt.loglog(t, np.rad2deg(pred), "--", lw=1.4, color=dark_col)

        # reference slope, anchored at the last point
        ref = np.rad2deg(rms[-1]) * (t / t[-1]) ** slope
        plt.text(t[-1]*0.74, ref[-1]*0.25, f"  $T^{{{slope}}}$", color=dark_col, fontsize=FONTSIZE_LEGEND, va="bottom")

    plt.xlabel("Time [seconds]",fontsize=FONTSIZE)
    plt.ylabel("Heading error [degrees, unwrapped]",fontsize=FONTSIZE)
    plt.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    plt.tick_params(axis="both", which="minor", labelsize=FONTSIZE)
    #plt.title("Single error sources")
    plt.grid(alpha=0.3, which="both")
    plt.legend(fontsize=FONTSIZE_LEGEND)
    plt.tight_layout()
    plt.savefig("gyro_validation_single_contributions.png", dpi=600)

    # FIGURE 2
    plt.figure()
    rms, pred, err = run_MC_simulations(SCENARIOS["all sources"])
    #plt.loglog(t, np.rad2deg(np.median(np.abs(err), axis=0)), lw=1.0, color="0.5",label="median")
    plt.loglog(t, np.rad2deg(rms), lw=2.0, color="C0", label="Monte Carlo simulation")
    plt.loglog(t, np.rad2deg(pred), "--", lw=2.0, color=dark_col, label="Analytical model")
    plt.xlabel("Time [seconds]",fontsize=FONTSIZE)
    plt.ylabel("Heading error [degrees, unwrapped]",fontsize=FONTSIZE)
    plt.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    plt.tick_params(axis="both", which="minor", labelsize=FONTSIZE)
    #plt.title("All error sources")
    plt.grid(alpha=0.3, which="both")
    plt.legend(fontsize=FONTSIZE_LEGEND)


    plt.tight_layout()
    
    plt.savefig("gyro_validation_all.png", dpi=600)

    # ---- Numerical check --------------------------------------------------
    ratio = rms / pred
    print(f"MC / model ratio:  min {ratio.min():.3f}   "
          f"max {ratio.max():.3f}   final {ratio[-1]:.3f}")
    for name in EXPECTED_SLOPE:
        r, p, _ = run_MC_simulations(SCENARIOS[name], NUMBER_OF_RUNS=200)
        # empirical slope from the last decade
        i0 = len(t) // 3
        slope = np.polyfit(np.log(t[i0:]), np.log(r[i0:]), 1)[0]
        print(f"{name:20s} empirical slope {slope:.2f} "
              f"(expected {EXPECTED_SLOPE[name]})")

    plt.show()