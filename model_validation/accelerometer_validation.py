import numpy as np
import matplotlib.pyplot as plt
from digitalshadow.devices.Floater import *

plt.style.use('seaborn-v0_8-deep')

NUMBER_OF_RUNS = 200
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
    "white noise only": 1.5,
    "random walk only": 2.5,
    "constant bias only": 2.0,
}


def build_floater(seed, cfg):
    """Create a static floater with only the selected error sources enabled."""
    np.random.seed(seed)
    f = Floater(0, 0, 0, 0, 1, 1.0, imu_seed=seed)

    # Static: no motion at all, so L = 0 and the heading term drops out
    f.set_initial_velocity(0.0, 0.0, 0.0)
    f.Rho = 1
    f.set_sigma(0.0, 0.0, 0.0)

    zero = np.zeros(3)

    # Simulator side: silence the sources that are off
    if not cfg["bias"]:
        f.accel_bias = zero.copy()
    if not cfg["white"]:
        f.sigma_white_noise = zero.copy()
    if not cfg["rw"]:
        f.sigma_bias_driving = zero.copy()

    # Estimator side: the analytical model must see the same configuration
    f.sigma_accel_bias = f.sigma_accel_bias if cfg["bias"] else zero.copy()

    # No attitude error in the static validation
    f.gyro_bias = 0.0
    f.sigma_gyro_bias = 0.0
    f.sigma_gyro_white_noise = 0.0
    f.sigma_gyro_bias_driving = 0.0
    f.sigma_yaw_rate = 0.0

    return f


def run_MC_simulations(cfg, NUMBER_OF_RUNS=NUMBER_OF_RUNS, steps=STEPS):
    err = np.zeros((NUMBER_OF_RUNS, steps))
    pred = np.zeros(steps) # Positioning error predicted by the analytical model

    for r in range(NUMBER_OF_RUNS):
        f = build_floater(r, cfg)
        for k in range(steps):
            f.move()
            # Horizontal error only
            err[r, k] = np.linalg.norm(f.est_pos[:2] - f.gt_pos[:2])

            # Analytical model is deterministic: we compute it only at the first run
            if r == 0:
                pred[k] = f.get_accumulated_error()

    # The model predicts the RMS of the 2-D error norm:
    # E[|r|^2] = sigma_x^2 + sigma_y^2 = sigma_pos^2

    # Returns: RMSE (gt vs est), analytical model, the series of error for each simulation
    return np.sqrt(np.mean(err**2, axis=0)), pred, err


if __name__ == "__main__":
    t = np.arange(1, STEPS + 1)

    # FIGURE 1
    for name, slope in EXPECTED_SLOPE.items():
        rms, pred, _ = run_MC_simulations(SCENARIOS[name])
        plt.loglog(t, rms, lw=1.8, label=f"{name} (MC)")
        dark_col = "#343434"
        plt.loglog(t, pred, "--", lw=1.4, color=dark_col)

        # reference slope, anchored at the last point
        ref = rms[-1] * (t / t[-1]) ** slope
        if slope == 2.0:
            yoff = 0.3
        elif slope == 2.5:
            yoff = 0.3
        else:
            yoff = 0.5
        plt.text(t[-1]*0.74, ref[-1]*yoff, f"  $T^{{{slope}}}$", color="0.4", fontsize=9,va="center")

    plt.xlabel("Time [seconds]", fontsize=FONTSIZE)
    plt.ylabel("Positioning error [meters]",fontsize=FONTSIZE)
    plt.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    plt.tick_params(axis="both", which="minor", labelsize=FONTSIZE)
    #plt.title("Single error sources",fontsize=FONTSIZE)
    plt.grid(alpha=0.3, which="both")
    plt.legend(fontsize=FONTSIZE_LEGEND)
    plt.tight_layout()
    plt.savefig("accel_validation_single_contributions.png", dpi=600)
    

    # ---- Panel 2: all sources together ------------------------------------
    plt.figure()
    rms, pred, err = run_MC_simulations(SCENARIOS["all sources"])
    #plt.loglog(t, np.median(err, axis=0), lw=1.0, color="0.5",label="median")
    plt.loglog(t, rms, lw=2.0, color="C0", label="Monte Carlo simulation")
    plt.loglog(t, pred, "--", lw=2.0, color=dark_col, label="Analytical model")

    plt.xlabel("Time [seconds]",fontsize=FONTSIZE)
    plt.ylabel("Positioning error [meters]",fontsize=FONTSIZE)
    #plt.title(f"All error sources",fontsize=FONTSIZE)
    plt.grid(alpha=0.3, which="both")
    plt.legend(fontsize=FONTSIZE_LEGEND)
    plt.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    plt.tick_params(axis="both", which="minor", labelsize=FONTSIZE)

    plt.tight_layout()
    plt.savefig("accel_validation_all.png", dpi=600)

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