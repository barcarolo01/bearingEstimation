import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# --- Stile condiviso con heatmap_plot.py (tenere allineati) ---
FONTSIZE = 20
FIGSIZE = (12, 10)
DPI = 900
CMAP = "plasma"
# --------------------------------------------------------------


def bearing_accuracy_map(floater_xy, sigma_deg, xlim, ylim, n=300):
    """CRLB della localizzazione da bearing su una griglia.
    Ritorna X, Y e, per ogni punto: DRMS, semiasse maggiore e minore (1σ) [m]."""
    sigma = np.deg2rad(sigma_deg)
    X, Y = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    J = np.zeros(X.shape + (2, 2))
    for fx, fy in np.asarray(floater_xy)[:, :2]:
        dx, dy = X - fx, Y - fy
        r2 = np.maximum(dx**2 + dy**2, 1e-6)
        g = np.stack([dy, -dx], axis=-1) / r2[..., None]
        J += g[..., :, None] * g[..., None, :] / sigma**2
    eig = np.clip(np.linalg.eigvalsh(J), 1e-30, None)   # autovalori crescenti
    var = 1.0 / eig                                     # autovalori di J^-1
    sig_major, sig_minor = np.sqrt(var[..., 0]), np.sqrt(var[..., 1])
    drms = np.sqrt(var.sum(-1))
    return X, Y, drms, sig_major, sig_minor


if __name__ == "__main__":
    plt.rcParams.update({"font.size": FONTSIZE})

    floaters = np.asarray([[-100, 0, 10],
                           [100, 0, 10]])

    fig, ax = plt.subplots(figsize=FIGSIZE)

    X, Y, drms, smaj, smin = bearing_accuracy_map(floaters, 1.0, (-600, 600), (-600, 600))

    cs = ax.contourf(X, Y, drms, levels=np.logspace(0, 3, 13), norm=LogNorm(), cmap=CMAP)

    ax.plot(floaters[:, 0], floaters[:, 1], color="red", linewidth=2, alpha=0.5, zorder=4)
    for i in range(floaters.shape[0]):
        ax.text(floaters[i, 0], floaters[i, 1], f"{i+1}", color="white", weight="bold",
                fontsize=10, fontfamily="monospace", va="center", ha="center", zorder=5,
                bbox=dict(facecolor="red", edgecolor="black",
                          boxstyle="circle,pad=0.5", linewidth=1))

    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax.set_xlabel("Easting [meters]", fontsize=FONTSIZE)
    ax.set_ylabel("Northing [meters]", fontsize=FONTSIZE)

    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("Localization RMSE [m]", fontsize=FONTSIZE)
    cbar.ax.tick_params(which="both", labelsize=FONTSIZE)

    fig.tight_layout()                    # dopo la colorbar, come in heatmap_plot.py
    fig.savefig("analytical_heatmap.pdf")
    plt.show()