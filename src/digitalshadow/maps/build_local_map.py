import numpy as np
import matplotlib.pyplot as plt

from digitalshadow.maps.map_common import (
    Track,
    as_multi,
    as_points,
    normalize_tracks,
    valid_points,
)

FONTSIZE = 18

FLOATER_COLOR = "#FF0000"
TX_COLOR = "#FFD700"
EST_COLOR = "#00CC66"


def build_local_cartesian_map(
    floater_coordinates=None,
    TX_coordinates=None,
    estimated_vessel_coordinates=None,
    tracks=None,
    output_file="map_local.png",
    track_alpha=0.7,
    window_width_m=None,
    window_height_m=None,
    Win=None,
    LEGEND=True
):
    """
    Draw the map on a local cartesian plane and save it as an image.

    All inputs are ALREADY expressed in a local cartesian frame (meters),
    not in geographic coordinates. The image is always centered on the
    origin (0, 0) of that frame.

    Parameters
    ----------
    floater_coordinates : array-like (N, M, 2|3) or (M, 2|3), or None
        N = time steps, M = number of floaters, columns [x, y, (depth)].
        An (M, 2|3) array is treated as a single time step (N = 1).
        Each floater has its own trajectory: different floaters are never connected.
    TX_coordinates : array-like (K, 2|3), or None
        Columns [x, y, (depth)].
    estimated_vessel_coordinates : array-like (K, 2|3), or None
        Columns [x, y, (depth)].
    tracks : Track | dict | tuple | sequence of those, or None
        Generic series of points. Each element holds:
          - name  : str, label of the series
          - points: array (NUMBER_STEPS, 2|3) or (NUMBER_STEPS, M, 2|3), columns [x, y, (depth)]
          - color : str, hexadecimal color
        With the 3-dimensional shape, M independent series sharing name and color
        are drawn: points are connected along the step axis only, never across
        different indices.
    output_file : str
    track_alpha : float
        Transparency of the series in `tracks`.
    window_width_m, window_height_m : float, or None
        Window size in meters, centered on (0, 0). When None they are derived
        from the data so that every point is visible.
    Win : float, or None
        Half-size of the window in meters. When given, the view is forced to
        [-Win, Win] on both axes, overriding window_width_m / window_height_m.

    x = Easting [m], y = Northing [m]. Depth: the depth column is optional;
    -999 values are treated as missing.
    """

    # --- Normalize every input to a single [x, y, depth] layout ---
    xy_floaters = as_multi(floater_coordinates)                          # (N, M, 3)
    xy_tx = valid_points(as_points(TX_coordinates))                      # (K, 3)
    xy_estimated = valid_points(as_points(estimated_vessel_coordinates)) # (K, 3)
    xy_tracks = [(t, np.asarray(t.xyz, dtype=float))                     # each (N, M, 3)
                 for t in normalize_tracks(tracks)]

    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, linestyle="--", alpha=0.6, color="#cccccc")

    # --- TX points (yellow) ---
    if len(xy_tx) > 1:
        ax.plot(xy_tx[:, 0], xy_tx[:, 1], color=TX_COLOR, linewidth=3, alpha=0.7, zorder=1)
    for point in xy_tx:
        ax.plot(point[0], point[1], marker="o", color=TX_COLOR, markersize=10,
                markeredgecolor="black", zorder=3)

    # --- Estimated positions (green) ---
    if len(xy_estimated) > 1:
        ax.plot(xy_estimated[:, 0], xy_estimated[:, 1], color=EST_COLOR, linewidth=3,
                linestyle="--", alpha=0.7, zorder=2)
    for point in xy_estimated:
        ax.plot(point[0], point[1], marker="o", color=EST_COLOR, markersize=8,
                markeredgecolor="black", zorder=4)

    # --- Generic tracks: one polyline per index, never connected to each other ---
    for track, xy_multi in xy_tracks:
        if xy_multi.size == 0:
            continue
        for series_index in range(xy_multi.shape[1]):
            points = valid_points(xy_multi[:, series_index, :])
            if len(points) == 0:
                continue
            if len(points) > 1:
                ax.plot(points[:, 0], points[:, 1], color=track.color, linewidth=2,
                        alpha=track_alpha, zorder=2)
            ax.plot(points[:, 0], points[:, 1], marker="o", markersize=5, color=track.color,
                    linestyle="None", markeredgecolor="black", markeredgewidth=0.5,
                    alpha=track_alpha, zorder=4)

    # --- Floaters: dashed trajectory + label on the first known position ---
    if xy_floaters.size > 0:
        for floater_index in range(xy_floaters.shape[1]):
            trajectory = valid_points(xy_floaters[:, floater_index, :])
            if len(trajectory) == 0:
                continue

            if len(trajectory) > 1:
                ax.plot(trajectory[:, 0], trajectory[:, 1], color=FLOATER_COLOR,
                        linewidth=1.5, linestyle="--", alpha=0.5, zorder=4)

            first_point = trajectory[0]
            ax.text(
                first_point[0], first_point[1], f"{floater_index + 1}",
                color="white", weight="bold", fontsize=10, fontfamily="monospace",
                va="center", ha="center", zorder=5,
                bbox=dict(facecolor=FLOATER_COLOR, edgecolor="black",
                          boxstyle="circle,pad=0.2", linewidth=1.5),
            )

    # --- Window size (always symmetric around the origin) ---
    if Win is not None:
        Win = float(Win)
        if Win <= 0:
            raise ValueError("Win must be a positive number of meters.")
        window_width_m = window_height_m = 2.0 * Win
    elif window_width_m is None or window_height_m is None:
        all_xy = [xy_tx[:, :2], xy_estimated[:, :2]]
        for _, xy_multi in xy_tracks:
            if xy_multi.size > 0:
                all_xy.append(valid_points(xy_multi.reshape(-1, 3))[:, :2])
        if xy_floaters.size > 0:
            all_xy.append(valid_points(xy_floaters.reshape(-1, 3))[:, :2])
        all_xy = [a for a in all_xy if len(a) > 0]
        span = max(np.abs(np.vstack(all_xy)).max() * 2.2, 10.0) if all_xy else 100.0
        window_width_m = window_width_m if window_width_m is not None else span
        window_height_m = window_height_m if window_height_m is not None else span

    ax.set_xlim(-window_width_m / 2.0, window_width_m / 2.0)
    ax.set_ylim(-window_height_m / 2.0, window_height_m / 2.0)

    ax.set_xlabel("Easting [meters]", fontsize=FONTSIZE, fontweight="bold")
    ax.set_ylabel("Northing [meters]", fontsize=FONTSIZE, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")

    # --- Legend ---
    #ax.plot(0, 0, "kx", markersize=5, markeredgewidth=2, label="Origin (0, 0)")

    if xy_floaters.size > 0:
        ax.plot([], [], marker="s", color=FLOATER_COLOR, linestyle="None", label="Floaters")
    if len(xy_tx) > 0:
        ax.plot([], [], marker="o", color=TX_COLOR, linestyle="None", label="Ground truth")
    if len(xy_estimated) > 0:
        ax.plot([], [], marker="o", color=EST_COLOR, linestyle="None", label="Estimated positions")
    for track, xy_multi in xy_tracks:
        if xy_multi.size > 0:
            ax.plot([], [], color=track.color, linestyle="-", label=track.name)

    if LEGEND:
        ax.legend(
            loc="upper left",
            #bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            frameon=True, facecolor="white", edgecolor="grey", fontsize=FONTSIZE,
        )

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Local map saved in: {output_file}")

    return output_file