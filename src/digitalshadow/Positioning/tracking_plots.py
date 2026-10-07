import os
import numpy as np
import matplotlib.pyplot as plt

# Map style, shared with build_local_map
FONTSIZE = 18            # axis labels and tick numbers
FONTSIZE_LEGEND = 14     # legend
FLOATER_COLOR = "#FF0000"
TX_COLOR = "#FFD700"
EST_COLOR = "#00CC66"

METHODS = {                     # key in res: (label, color, line style)
    'raw':      ('Raw estimation',  EST_COLOR,  '--'),
    'filtered': ('Kalman filter',   '#F58518', '--'),
    'smoothed': ('Kalman + RTS',    '#3489eb', '--'),
}

# Common format of the three estimated trajectories
TRACK_STYLE = dict(linewidth=3, marker="o", markersize=7,
                   markeredgecolor="black", markeredgewidth=0.8)
TRACK_ALPHA = 0.8
GT_ALPHA = 0.35


def _active(res):
    """Methods present in res with a non-None value, in METHODS order."""
    return {k: v for k, v in METHODS.items() if res.get(k) is not None}


def _local(points):
    """[x, y, ...] -> [x, y] in metres in the local plane."""
    return np.asarray(points, dtype=float)[..., :2]


def _valid(xy):
    """Drops the rows containing NaN / inf."""
    xy = np.asarray(xy, dtype=float)
    return xy[np.all(np.isfinite(xy), axis=-1)]


def _errors(res, tx_gt):
    """Horizontal localization error [m] per step for every available method."""
    return {k: np.hypot(*(np.asarray(res[k], dtype=float)[..., :2] - tx_gt[..., :2]).T)
            for k in _active(res)}


def _save_paths(save_path):
    """'out.png' -> ('out_map.png', 'out_error.png'); a (map, error) pair is used as is."""
    if save_path is None:
        return None, None
    if isinstance(save_path, (tuple, list)):
        return tuple(save_path)
    root, ext = os.path.splitext(save_path)
    ext = ext or '.png'
    return f"{root}_map{ext}", f"{root}_error{ext}"


def tracking_report(res, tx_gt):
    """Prints RMSE, median, 90th percentile and max error for each available method."""
    err = _errors(res, tx_gt)
    print("===== Kalman filter report =====")
    print(f"{'method':14s} {'RMSE':>8s} {'median':>8s} {'p90':>8s} {'max':>8s} {'valid':>9s}")
    out = {}
    for k, e in err.items():
        v = e[np.isfinite(e)]
        if v.size == 0:
            continue
        out[k] = dict(rmse=np.sqrt(np.mean(v ** 2)), median=np.median(v),
                      p90=np.percentile(v, 90), max=v.max(), valid=v.size)
        m = out[k]
        print(f"{METHODS[k][0]:14s} {m['rmse']:8.1f} {m['median']:8.1f} {m['max']:8.1f} {m['valid']:4d}/{len(tx_gt):<4d}")
    return out


def plot_tracking_map(res, tx_gt, floaters, out_folder="", legend=True):
    """Map of the trajectories, in the same style as build_local_map."""
    floaters = np.asarray(floaters, dtype=float)
    gt = tx_gt[..., :2]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax.tick_params(axis="both", which="minor", labelsize=FONTSIZE)
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, linestyle="--", alpha=0.6, color="#cccccc")

    # --- Ground truth (semi-transparent yellow line + circles, in background) ---
    ax.plot(*gt.T, color=TX_COLOR, linewidth=3, marker="o", markersize=10,
            markeredgecolor="black", alpha=GT_ALPHA, zorder=1, label="Ground truth")

    # --- Estimates: same format, different color and line style (only those available) ---
    for z, (k, (label, color, ls)) in enumerate(_active(res).items()):
        ax.plot(*np.asarray(res[k], dtype=float)[..., :2].T, color=color, linestyle=ls,
                alpha=TRACK_ALPHA, zorder=2 + z, label=label, **TRACK_STYLE)

    # --- Floaters: trajectory + numbered label on the first known position ---
    for m in range(floaters.shape[1]):
        traj = _valid(floaters[:, m, :2])
        if len(traj) == 0:
            continue
        if len(traj) > 1:
            ax.plot(*traj.T, color=FLOATER_COLOR, linewidth=2, alpha=0.5, zorder=5)
        ax.text(traj[0, 0], traj[0, 1], f"{m + 1}",
                color="white", weight="bold", fontsize=10, fontfamily="monospace",
                va="center", ha="center", zorder=6,
                bbox=dict(facecolor=FLOATER_COLOR, edgecolor="black",
                          boxstyle="circle,pad=0.2", linewidth=1.5))
    ax.plot([], [], marker="o", color=FLOATER_COLOR, linestyle="None", label="Floaters")

    # Frame on ground truth and floaters, so that diverging raw fixes
    # near the baseline do not shrink the plot
    pts = _valid(np.vstack([gt, floaters.reshape(-1, floaters.shape[2])[..., :2]]))
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.25 * max(hi - lo)
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)

    ax.set_xlabel("Easting [meters]", fontsize=FONTSIZE, fontweight="bold")
    ax.set_ylabel("Northing [meters]", fontsize=FONTSIZE, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")

    if legend:
        ax.legend(loc="upper left", borderaxespad=0, frameon=True,
                  facecolor="white", edgecolor="grey", fontsize=FONTSIZE_LEGEND)

    fig.tight_layout()
    fig.savefig(os.path.join(out_folder,"map_plot.png"), dpi=900, bbox_inches="tight")
    plt.show()
    return fig


def plot_tracking_error(res, tx_gt, out_folder=''):
    """Localization error per step for every available method."""
    err = _errors(res, tx_gt)
    if 'filtered' not in err:
        print("Filtered trajectiory not present: tracking error plot rejected.")
        return None
    
    steps = np.arange(len(tx_gt))
    fig, ax = plt.subplots(figsize=(10, 10))
    has_data = False
    for k, e in err.items():
        label, color, ls = METHODS[k]
        ax.plot(steps, e, color=color, ls=ls, lw=3, marker='o', ms=4,
                markeredgecolor='black', markeredgewidth=0.5, label=label)
        has_data |= bool(np.any(np.isfinite(e) & (e > 0)))
    if has_data:
        ax.set_yscale('log')
    ax.tick_params(axis="both", which="minor", labelsize=FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax.set_xlabel('Simulation step', fontsize=FONTSIZE)
    ax.set_ylabel('Localization error [m]', fontsize=FONTSIZE)
    ax.grid(alpha=0.3, which='both')
    if err:
        ax.legend(fontsize=FONTSIZE_LEGEND)

    fig.tight_layout()
    fig.savefig(os.path.join(out_folder,"kalman_error.png"), dpi=900, bbox_inches='tight')

    return fig