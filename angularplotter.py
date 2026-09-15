import numpy as np
import matplotlib.pyplot as plt

STEPS, DT = 300, 1.0



#Variable parameters
var_sigmas = [0.2, 1.0, 5.0]
var_rhos = [0.5, 0.9, 0.99]

SIG_REF = 2.0
RHO_REF = var_rhos[1]

FONTSIZE = 12

def run(sig, rho, seed):
    rng = np.random.default_rng(seed)
    w, psi, hist = 0.0, 0.0, []
    for _ in range(STEPS):
        w = w*rho + rng.normal(0, np.deg2rad(sig))*np.sqrt(1-rho**2)
        psi += w*DT
        hist.append(np.rad2deg(psi))
    return hist

if __name__ == "__main__":
    fig, axes = plt.subplots(
        2, 3, figsize=(12, 6), sharex=True, sharey='row')

    # First row: variation of SIGMA
    for ax, sig in zip(axes[0], var_sigmas):
        for k in range(3):
            ax.plot(run(sig, RHO_REF, k+100), label=f"F{k+1}" )

        ax.set_title(rf"$\sigma_\omega = {sig}^\circ/s$",fontsize=FONTSIZE)

    # Second row: variation of RHO
    for ax, rho in zip(axes[1], var_rhos):
        for k in range(3):
            ax.plot(run(SIG_REF, rho, k+100),label=f"F{k+1}")
        ax.set_title(rf"$\rho_\psi = {rho}$",fontsize=FONTSIZE)

    # Formattazione generale
    for ax in axes.ravel():
        ax.axhline(0, color='k', lw=0.5)
        ax.grid(alpha=0.3)

        # Linee tratteggiate ogni multiplo di 360°
        ymin, ymax = ax.get_ylim()

        primo_multiplo = int(np.ceil(ymin / 360))
        ultimo_multiplo = int(np.floor(ymax / 360))

        for n in range(primo_multiplo, ultimo_multiplo + 1):
            if n != 0:
                ax.axhline(
                    n * 360,
                    color='k',
                    linestyle='--',
                    linewidth=0.5,
                    alpha=0.5
                )

    # Etichette asse X
    for ax in axes[1]:
        ax.set_xlabel("Simulation step",fontsize=FONTSIZE)

    # Etichette asse Y
    axes[0, 0].set_ylabel(r"$\psi$ [deg]", fontsize=FONTSIZE)

    axes[1, 0].set_ylabel(r"$\psi$ [deg]", fontsize=FONTSIZE)

    # Legend
    axes[0, 0].legend(fontsize=FONTSIZE)

    fig.text(0.97, 0.735,
        rf"$\rho_\psi$ = {RHO_REF}",
        fontsize=FONTSIZE,ha='right',va='center')

    fig.text(0.99, 0.265,
        rf"$\sigma$ = {SIG_REF}°/s",
        fontsize=FONTSIZE,ha='right',va='center')

    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.savefig("floater_mobility_gyro.png",dpi = 300)
    plt.show()