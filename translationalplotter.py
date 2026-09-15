import numpy as np
import matplotlib.pyplot as plt

STEPS, DT = 300, 1.0

# Variable parameters
var_sigmas = [0.05, 0.1, 1.0]          # [m/s]
var_rhos   = [0.999, 0.99, 0.9]

SIG_REF = 0.2
RHO_REF = 0.99

# Initial velocity of each floater: (vx, vy) in m/s
V0 = [(0.0, 0.5), (0.5, 0.0), (0.25, 0.25)]

FONTSIZE = 12


def run(sig, rho, seed, v0):
    """AR(1) on the velocity components, integrated to obtain the trajectory."""
    rng = np.random.default_rng(seed)
    v = np.array(v0, dtype=float)
    s = np.zeros(2)
    hist = [s.copy()]
    for _ in range(STEPS):
        e = rng.normal(0, sig, size=2)
        v = v * rho + e * np.sqrt(1 - rho**2)
        s = s + v * DT
        hist.append(s.copy())
    return np.asarray(hist)


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # First row: variation of SIGMA
    for ax, sig in zip(axes[0], var_sigmas):
        for k in range(3):
            traj = run(sig, RHO_REF, k + 100, V0[k])
            ax.plot(traj[:, 0], traj[:, 1], lw=1.2, label=f"F{k+1}")
        ax.set_title(rf"$\sigma = {sig}$ m/s", fontsize=FONTSIZE)

    # Second row: variation of RHO
    for ax, rho in zip(axes[1], var_rhos):
        for k in range(3):
            traj = run(SIG_REF, rho, k + 100, V0[k])
            ax.plot(traj[:, 0], traj[:, 1], lw=1.2, label=f"F{k+1}")
        ax.set_title(rf"$\rho = {rho}$", fontsize=FONTSIZE)

    # Formattazione generale
    for ax in axes.ravel():
        ax.plot(0, 0, marker='o', ms=5, color='k', zorder=5)
        ax.set_aspect('equal', adjustable='datalim')
        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.axvline(0, color='k', lw=0.5, alpha=0.5)
        ax.grid(alpha=0.3)

    # Etichette assi
    for ax in axes[1]:
        ax.set_xlabel("Easting [m]", fontsize=FONTSIZE)
    axes[0, 0].set_ylabel("Northing [m]", fontsize=FONTSIZE)
    axes[1, 0].set_ylabel("Northing [m]", fontsize=FONTSIZE)

    # Legend
    axes[0, 0].legend(fontsize=FONTSIZE)

    fig.text(0.97, 0.735,
             rf"$\rho$ = {RHO_REF}",
             fontsize=FONTSIZE, ha='right', va='center')

    fig.text(0.99, 0.265,
             rf"$\sigma$ = {SIG_REF} m/s",
             fontsize=FONTSIZE, ha='right', va='center')

    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.savefig("floater_mobility_translational.png", dpi=300)
    plt.show()