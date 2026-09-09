"""Heatmap of localization error, with the static receivers overlaid."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, to_rgb

CSV_PATH = "accuracy.csv"
RX_PATH = "RX_static_coordinates.npy"   # [N, 3] -> Lat, Lon, Depth
FONTSIZE = 14
CMAP = "inferno"
NAN_COLOR = "white"
MODE = "nearest"      # imshow interpolation: "nearest", "bilinear", "bicubic", ...
R_EARTH = 6378137.0

from scipy.interpolate import griddata


def to_local_en(lat, lon, lat0, lon0):
    """Geodetic coordinates -> local East/North plane [m] centred on (lat0, lon0)."""
    east = np.radians(lon - lon0) * R_EARTH * np.cos(np.radians(lat0))
    north = np.radians(lat - lat0) * R_EARTH
    return east, north

def grid_extent(xs, ys):
    """Cell centres -> imshow extent, padded by half a cell on each side."""
    dx = xs[1] - xs[0] if xs.size > 1 else 1.0
    dy = ys[1] - ys[0] if ys.size > 1 else 1.0
    return [xs[0] - dx / 2, xs[-1] + dx / 2, ys[0] - dy / 2, ys[-1] + dy / 2]


def make_norm(values, title):
    """Log scale when possible, linear otherwise, None when there is nothing to plot."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        print(f"[WARN] '{title}': no finite value, figure skipped.")
        return None

    positive = finite[finite > 0]
    if positive.size:
        return LogNorm(vmin=positive.min(), vmax=max(positive.max(), positive.min() * 10))

    print(f"[INFO] '{title}': no positive value, linear scale used.")
    return Normalize(vmin=finite.min(), vmax=max(finite.max(), finite.min() + 1e-9))


"""Heatmap of localization error, with the static receivers overlaid."""


def fill_holes(xs, ys, grid):
    """Interpolate the empty cells so the image has no masked region left."""
    holes = np.isnan(grid)
    if not holes.any():
        return grid

    X, Y = np.meshgrid(xs, ys)
    ok = ~holes
    pts, vals = np.column_stack((X[ok], Y[ok])), grid[ok]

    out = griddata(pts, vals, (X, Y), method="linear")
    outside = np.isnan(out)          # cells beyond the convex hull of the samples
    if outside.any():
        out[outside] = griddata(pts, vals, (X[outside], Y[outside]), method="nearest")
    return out


def to_grid(east, north, values, decimals=1):
    """Regularly-spaced samples -> (x_centres, y_centres, 2-D array with NaN gaps)."""
    e, n = np.round(east, decimals), np.round(north, decimals)
    xs, ys = np.unique(e), np.unique(n)
    grid = np.full((ys.size, xs.size), np.nan)
    grid[np.searchsorted(ys, n), np.searchsorted(xs, e)] = values
    return xs, ys, grid

def plot_heatmap(east, north, values, rx_east, rx_north, title, cbar_label, out_path):
    norm = make_norm(values, title)
    if norm is None:
        return

    xs, ys, grid = to_grid(east, north, values)
    holes = np.isnan(grid)              # maschera: va calcolata PRIMA di riempire
    filled = fill_holes(xs, ys, grid)   # griglia senza NaN, per l'immagine di fondo

    if holes.any():
        print(f"[INFO] '{title}': {int(holes.sum())} empty cell(s), blended in {NAN_COLOR}.")

    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_under(NAN_COLOR)   # non-positive values (invisible on a log scale)

    ext = grid_extent(xs, ys)

    fig, ax = plt.subplots(figsize=(9.0, 7.5))
    ax.set_facecolor(NAN_COLOR)
    mesh = ax.imshow(filled, extent=ext, origin="lower",
                     cmap=cmap, norm=norm, interpolation=MODE, aspect="equal")

    # Empty cells: opaque overlay whose alpha is interpolated -> soft edges
    if holes.any():
        overlay = np.ones(grid.shape + (3,)) * to_rgb(NAN_COLOR)
        ax.imshow(overlay, extent=ext, origin="lower",
                  alpha=holes.astype(float), interpolation=MODE,
                  aspect="equal", zorder=2)

    # Receivers: red dot with its label inside
    ax.scatter(rx_east, rx_north, s=420, c="red",
               edgecolors="white", linewidths=1.2, zorder=3)
    for k, (e, n) in enumerate(zip(rx_east, rx_north), start=1):
        ax.text(e, n, f"F{k}", ha="center", va="center", color="white",
                fontsize=FONTSIZE - 3, fontweight="bold", zorder=4)

    fig.colorbar(mesh, ax=ax).set_label(cbar_label)
    ax.set_xlabel("Easting [meters]")
    ax.set_ylabel("Northing [meters]")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    print(f"[OK] saved: {out_path}")

def main():
    plt.rcParams.update({"font.size": FONTSIZE})

    df = pd.read_csv(CSV_PATH).dropna(subset=["GT_Lat", "GT_Lon", "GT_Depth"])
    lat0, lon0 = df["GT_Lat"].mean(), df["GT_Lon"].mean()

    gt_e, gt_n = to_local_en(df["GT_Lat"].values, df["GT_Lon"].values, lat0, lon0)
    est_e, est_n = to_local_en(df["Est_Lat"].values, df["Est_Lon"].values, lat0, lon0)

    rx = np.load(RX_PATH)
    rx_e, rx_n = to_local_en(rx[:, 0], rx[:, 1], lat0, lon0)

    # Planar map: always produced.
    planar_error = np.hypot(est_e - gt_e, est_n - gt_n)
    plot_heatmap(gt_e, gt_n, planar_error, rx_e, rx_n,
                 "Planar localization error", "Planar error [m]",
                 "planar_error_heatmap.png")

    # Depth map: only if at least one depth estimate is available.
    est_depth = pd.to_numeric(df["Est_depth"], errors="coerce").values
    n_valid = int(np.isfinite(est_depth).sum())
    if n_valid:
        print(f"[INFO] depth map: {n_valid}/{len(est_depth)} valid estimate(s).")
        depth_error = np.abs(df["GT_Depth"].values - est_depth)
        plot_heatmap(gt_e, gt_n, depth_error, rx_e, rx_n,
                     "Depth error", "Depth error [m]",
                     "depth_error_heatmap.png")
    else:
        print("[INFO] no valid depth estimate, depth map skipped.")

    plt.show()



def main():
    plt.rcParams.update({"font.size": FONTSIZE})

    df = pd.read_csv(CSV_PATH).dropna(subset=["GT_Lat", "GT_Lon", "GT_Depth"])
    lat0, lon0 = df["GT_Lat"].mean(), df["GT_Lon"].mean()

    gt_e, gt_n = to_local_en(df["GT_Lat"].values, df["GT_Lon"].values, lat0, lon0)
    est_e, est_n = to_local_en(df["Est_Lat"].values, df["Est_Lon"].values, lat0, lon0)

    rx = np.load(RX_PATH)
    rx_e, rx_n = to_local_en(rx[:, 0], rx[:, 1], lat0, lon0)

    # Planar map: always produced.
    planar_error = np.hypot(est_e - gt_e, est_n - gt_n)
    plot_heatmap(gt_e, gt_n, planar_error, rx_e, rx_n,
                 "Planar localization error", "Planar error [m]",
                 "planar_error_heatmap.png")

    # Depth map: only if at least one depth estimate is available.
    est_depth = pd.to_numeric(df["Est_depth"], errors="coerce").values
    n_valid = int(np.isfinite(est_depth).sum())
    if n_valid:
        print(f"[INFO] depth map: {n_valid}/{len(est_depth)} valid estimate(s).")
        depth_error = np.abs(df["GT_Depth"].values - est_depth)
        plot_heatmap(gt_e, gt_n, depth_error, rx_e, rx_n,
                     "Depth error", "Depth error [m]",
                     "depth_error_heatmap.png")
    else:
        print("[INFO] no valid depth estimate, depth map skipped.")

    plt.show()


if __name__ == "__main__":
    main()