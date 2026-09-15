import os
import numpy as np
import matplotlib.pyplot as plt

from maps.map_common import (
    Track,
    as_multi,
    as_points,
    first_valid_location,
    normalize_tracks,
    valid_points,
)

FONTSIZE = 18

FLOATER_COLOR = "#FF0000"
TX_COLOR = "#FFD700"
EST_COLOR = "#00CC66"

# Image formats handled by matplotlib (the local map cannot produce HTML)
SUPPORTED_EXTENSIONS = {
    "png", "pdf", "svg", "svgz", "eps", "ps", "jpg", "jpeg",
    "tif", "tiff", "webp", "raw", "rgba", "pgf",
}


def build_local_cartesian_map(
    floater_coordinates=None,
    TX_coordinates=None,
    estimated_vessel_coordinates=None,
    tracks=None,
    output_file="map_local.png",
    track_alpha=0.7,
    center_coordinates=None,
    window_width_m=None,
    window_height_m=None,
):
    """
    Draw the map on a local cartesian plane (equirectangular projection centered
    on `center_coordinates`) and save it as an image.

    Parameters shared with `build_folium_map`
    -----------------------------------------
    floater_coordinates : array-like (N, M, 2|3) or (M, 2|3), or None
        N = time steps, M = number of floaters, columns [lat, lon, (depth)].
        An (M, 2|3) array is treated as a single time step (N = 1).
        Each floater has its own trajectory: different floaters are never connected.
    TX_coordinates : array-like (K, 2|3), or None
    estimated_vessel_coordinates : array-like (K, 2|3), or None
    tracks : Track | dict | tuple | sequence of those, or None
        Generic series of points. Each element holds:
          - name  : str, label of the series
          - points: array (NUMBER_STEPS, 2|3) or (NUMBER_STEPS, M, 2|3)
          - color : str, hexadecimal color
        With the 3-dimensional shape, M independent series sharing name and color
        are drawn: points are connected along the step axis only, never across
        different indices.
    output_file : str
    track_alpha : float
        Transparency of the series in `tracks`.

    Parameters specific to the local map
    ------------------------------------
    center_coordinates : (lat, lon), or None
        Origin of the local frame. When None, the first available valid point is
        used (floaters -> TX -> estimated -> tracks).
    window_width_m, window_height_m : float, or None
        Window size in meters. When None they are derived from the data.

    Depth: the depth column is optional; -999 values are treated as missing.
    """

    # --- Output format check ---
    ext = os.path.splitext(str(output_file))[1].lower().lstrip(".")
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"'{output_file}': format '{ext or 'missing'}' is not supported by the local map. "
            f"Use one of these extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}. "
            "HTML files must be passed to build_folium_map()."
        )

    # --- Normalize every input to a single [lat, lon, depth] layout ---
    floaters = as_multi(floater_coordinates)                    # (N, M, 3)
    tx = as_points(TX_coordinates)                              # (K, 3)
    estimated = as_points(estimated_vessel_coordinates)         # (K, 3)
    track_list = normalize_tracks(tracks)                       # each .xyz is (N, M, 3)

    if center_coordinates is None:
        center_coordinates = first_valid_location(
            floaters, tx, estimated, *[t.xyz for t in track_list]
        )
        if center_coordinates is None:
            raise ValueError("No valid coordinate available to center the map.")

    EARTH_RADIUS = 6371000.0
    lat_ref = np.radians(float(center_coordinates[0]))
    lon_ref = np.radians(float(center_coordinates[1]))

    def _geo_to_local(arr):
        """[lat, lon, depth] -> [x, y, depth] in meters relative to the center."""
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return np.empty(arr.shape)
        orig_shape = arr.shape
        flat = arr.reshape(-1, orig_shape[-1])
        x = EARTH_RADIUS * (np.radians(flat[:, 1]) - lon_ref) * np.cos(lat_ref)
        y = EARTH_RADIUS * (np.radians(flat[:, 0]) - lat_ref)
        out = np.column_stack((x, y, flat[:, 2]))
        return out.reshape(orig_shape)

    xy_floaters = _geo_to_local(floaters)                       # (N, M, 3)
    xy_tx = valid_points(_geo_to_local(tx))                     # (K, 3)
    xy_estimated = valid_points(_geo_to_local(estimated))       # (K, 3)
    xy_tracks = [(t, _geo_to_local(t.xyz)) for t in track_list]

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
            # series `series_index`, in step order
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

    # --- Window size ---
    if window_width_m is None or window_height_m is None:
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
    center_label = (f"Center coordinates\n"
                    f"[{float(center_coordinates[0]):.5f}, {float(center_coordinates[1]):.5f}]")
    ax.plot(0, 0, "kx", markersize=5, markeredgewidth=2, label=center_label)

    if xy_floaters.size > 0:
        ax.plot([], [], marker="s", color=FLOATER_COLOR, linestyle="None", label="Floaters")
    if len(xy_tx) > 0:
        ax.plot([], [], marker="o", color=TX_COLOR, linestyle="None", label="Ground truth")
    if len(xy_estimated) > 0:
        ax.plot([], [], marker="o", color=EST_COLOR, linestyle="None", label="Estimated positions")
    for track, xy_multi in xy_tracks:
        if xy_multi.size > 0:
            ax.plot([], [], color=track.color, linestyle="-", label=track.name)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=True, facecolor="white", edgecolor="grey", fontsize=FONTSIZE,
    )


    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Local map saved in: {output_file}")

    return output_file


# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    RX_Coords = np.load("Synth/RX_Coordinates.npy")
    TX_Coords = np.load("Synth/TX_Coordinates.npy")
    Est_Coords = np.load("Synth/Estimated_Coordinates.npy")

    build_local_cartesian_map(
        floater_coordinates=RX_Coords,
        TX_coordinates=TX_Coords,
        estimated_vessel_coordinates=Est_Coords,
        tracks=[
            Track("RX IMU", np.load("Synth/RX_fw_IMU.npy"), "#0000FF"),
            Track("RX IMU+MDS", np.load("Synth/RX_fw_IMU_MDS.npy"), "#2AB040"),
            Track("Compensated", np.load("Synth/RX_bw_IMU.npy"), "#FF8822"),
        ],
        output_file="map_local.png",
        # None -> automatically centered on the first valid point
        center_coordinates=RX_Coords.reshape(-1, RX_Coords.shape[-1])[0, :2],
        window_width_m=40,
        window_height_m=40,
    )
