import numpy as np
import matplotlib.pyplot as plt

METHODS = {                     # key in res: (label, color, line style)
    'raw':      ('Raw estimation',     '#B279A2', ''),
    'filtered': ('Kalman filter',   '#F58518', '-'),
    'smoothed': ('Kalman + RTS',    '#54A24B', '--'),
}

def _local(points):
    """[x, y, ...] -> [x, y] in metres in the local plane."""
    return np.asarray(points, dtype=float)[..., :2]

def _errors(res, tx_gt):
    """Horizontal localization error [m] per step for every method."""
    return {k: np.hypot(*(res[k][..., :2] - tx_gt[..., :2]).T) for k in METHODS}


def tracking_report(res, tx_gt):
    """Prints RMSE, median, 90th percentile and max error for each method."""
    err = _errors(res, tx_gt)
    N = len(tx_gt)
    print(f"{'method':14s} {'RMSE':>8s} {'median':>8s} {'p90':>8s} {'max':>8s} {'valid':>9s}")
    out = {}
    for k, e in err.items():
        v = e[np.isfinite(e)]
        if v.size == 0:
            continue
        out[k] = dict(rmse=np.sqrt(np.mean(v ** 2)), median=np.median(v),
                      p90=np.percentile(v, 90), max=v.max(), valid=v.size)
        m = out[k]
        print(f"{METHODS[k][0]:14s} {m['rmse']:8.1f} {m['median']:8.1f} {m['p90']:8.1f} "
              f"{m['max']:8.1f} {m['valid']:4d}/{N:<4d}")
    return out


def plot_tracking(res, tx_gt, floaters, save_path=None):
    """Map of the trajectories (left) and localization error per step (right)."""
    floaters = np.asarray(floaters, dtype=float)
    gt = tx_gt[..., :2]
    err = _errors(res, tx_gt)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.5),
                                  gridspec_kw=dict(width_ratios=[1, 1.2]))

    # --- Map -------------------------------------------------------------
    ax.plot(*gt.T, color='0.55', lw=2, label='Ground truth', zorder=1)
    for k, (label, color, ls) in METHODS.items():
        xy = res[k][..., :2]
        if k == 'raw':
            ax.scatter(*xy.T, s=16, marker='x', color=color, label=label, zorder=2)
        else:
            ax.plot(*xy.T, color=color, lw=1.8, ls=ls, label=label, zorder=3)
    for m in range(floaters.shape[1]):
        fxy = floaters[:, m][..., :2]
        ax.plot(*fxy.T, color='k', lw=0.8)                       # drift, if any
        ax.scatter(*fxy[-1], marker='^', s=80, color='k', zorder=4,
                   label='Floaters' if m == 0 else None)

    # Frame on ground truth and floaters, so that diverging raw fixes
    # near the baseline do not shrink the plot
    pts = np.vstack([gt, floaters.reshape(-1, floaters.shape[2])[..., :2]])
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.25 * max(hi - lo)
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    # --- Error per step ----------------------------------------------------
    steps = np.arange(len(gt))
    for k, (label, color, ls) in METHODS.items():
        if k == 'raw':
            ax2.plot(steps, err[k], color=color, lw=1.0, marker='x', ms=4, label=label)
        else:
            ax2.plot(steps, err[k], color=color, lw=1.8, ls=ls, label=label)
    ax2.set_yscale('log')
    ax2.set_xlabel('Simulation step')
    ax2.set_ylabel('Localization error [m]')
    ax2.grid(alpha=0.3, which='both')
    ax2.legend(fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    return fig