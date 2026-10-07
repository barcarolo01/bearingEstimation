import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Any, Optional

# Sentinel value used in the arrays for "depth not available"
NODATA_DEPTH = -999.0

# Colors assigned in round-robin fashion when a track does not specify one
DEFAULT_TRACK_COLORS = [
    "#0000FF",  # blue
    "#FF8822",  # orange
    "#2AB040",  # green
    "#AA00AA",  # purple
    "#00AACC",  # cyan
    "#884400",  # brown
]


@dataclass
class Track:
    """
    Series of points to be displayed on the map.

    Parameters
    ----------
    name : str
        Label of the series (legend on matplotlib, tooltip/popup on folium).
    points : array-like of shape (NUMBER_STEPS, 2|3) or (NUMBER_STEPS, M, 2|3)
        Points as [lat, lon] or [lat, lon, depth]. With the 3-dimensional shape,
        M independent series are drawn (one per index); they share name and color
        but are never connected to each other.
        NaN values and -999 depths are treated as missing data.
    color : str
        Hexadecimal color, e.g. "#0000FF".
    """

    name: str
    points: Any
    color: str = DEFAULT_TRACK_COLORS[0]
    # filled in by `normalize_tracks`: array (N, M, 3) [step, series, lat/lon/depth]
    xyz: Optional[np.ndarray] = field(default=None, repr=False, compare=False)


def is_valid(*values) -> bool:
    """True only if none of the values is None or NaN."""
    return all(v is not None and not np.isnan(float(v)) for v in values)


def _as_float_array(arr) -> Optional[np.ndarray]:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    if a.size == 0 or a.ndim < 2 or a.shape[-1] < 2:
        return None
    return a


def _with_depth(flat: np.ndarray) -> np.ndarray:
    """(K, C) -> (K, 3), with depth = NaN when missing or equal to NODATA_DEPTH."""
    if flat.shape[-1] >= 3:
        depth = np.where(flat[:, 2] == NODATA_DEPTH, np.nan, flat[:, 2])
    else:
        depth = np.full(flat.shape[0], np.nan)
    return np.column_stack((flat[:, 0], flat[:, 1], depth))


def as_points(arr) -> np.ndarray:
    """
    Any array of shape (..., 2|3) -> (K, 3) with columns [lat, lon, depth].
    Returns an empty (0, 3) array when `arr` is None or empty.
    """
    a = _as_float_array(arr)
    if a is None:
        return np.empty((0, 3))
    return _with_depth(a.reshape(-1, a.shape[-1]))


def as_multi(arr) -> np.ndarray:
    """
    (N, M, 2|3) -> (N, M, 3). An (M, 2|3) array is treated as a single time step
    (N = 1). Returns an empty (0, 0, 3) array when `arr` is None.
    """
    a = _as_float_array(arr)
    if a is None:
        return np.empty((0, 0, 3))
    if a.ndim == 2:
        a = a[np.newaxis, :, :]
    n, m, c = a.shape[0], a.shape[1], a.shape[-1]
    return _with_depth(a.reshape(-1, c)).reshape(n, m, 3)


def as_series(arr) -> np.ndarray:
    """
    Normalize the points of a Track to shape (N, M, 3) = [step, series, lat/lon/depth].

    (N, 2|3)     -> (N, 1, 3): a single series of N steps.
    (N, M, 2|3)  -> (N, M, 3): M independent series of N steps each.

    Unlike `as_multi` (used for the floaters), a 2D array is interpreted as a
    time sequence, not as a single multi-device time step.
    """
    a = _as_float_array(arr)
    if a is None:
        return np.empty((0, 0, 3))
    if a.ndim == 2:
        a = a[:, np.newaxis, :]
    n, m, c = a.shape[0], a.shape[1], a.shape[-1]
    return _with_depth(a.reshape(-1, c)).reshape(n, m, 3)


def valid_points(points: np.ndarray) -> np.ndarray:
    """Keep only the (K, 3) rows whose lat/lon are finite."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.empty((0, pts.shape[-1] if pts.ndim > 1 else 3))
    mask = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    return pts[mask]


def depth_str(depth) -> str:
    """Format a depth value for popups and labels."""
    return f"{depth:.1f} m" if depth is not None and np.isfinite(depth) else "N/A"


def _to_track(item, index: int) -> Track:
    """Convert a Track / dict / (name, points[, color]) tuple into a normalized Track."""
    default_color = DEFAULT_TRACK_COLORS[index % len(DEFAULT_TRACK_COLORS)]

    if isinstance(item, Track):
        name, points, color = item.name, item.points, item.color
    elif isinstance(item, dict):
        name = item.get("name", f"Track {index + 1}")
        points = item.get("points", item.get("coordinates"))
        color = item.get("color", default_color)
    elif isinstance(item, (tuple, list)) and 2 <= len(item) <= 3:
        name, points = item[0], item[1]
        color = item[2] if len(item) == 3 else default_color
    else:
        raise TypeError(
            "Every element of `tracks` must be a Track, a dict "
            "{'name', 'points', 'color'} or a (name, points[, color]) tuple; "
            f"got {type(item)!r}."
        )

    return Track(name=str(name), points=points, color=color, xyz=as_series(points))


def normalize_tracks(tracks) -> list:
    """
    Normalize the `tracks` parameter into a list of Track objects whose `xyz`
    field holds an (N, M, 3) array. Accepts None, a single Track/dict/tuple,
    or a sequence of those.
    """
    if tracks is None:
        return []
    if isinstance(tracks, (Track, dict)):
        tracks = [tracks]
    elif isinstance(tracks, (tuple, list)) and len(tracks) >= 2 and isinstance(tracks[0], str):
        # a single (name, points[, color]) tuple passed directly
        tracks = [tracks]
    return [_to_track(t, i) for i, t in enumerate(tracks)]


def first_valid_location(*point_arrays):
    """First valid (lat, lon) point among the given arrays, in order of priority."""
    for pts in point_arrays:
        if pts is None:
            continue
        pts = np.asarray(pts, dtype=float)
        if pts.size == 0:
            continue
        flat = pts.reshape(-1, pts.shape[-1])
        valid = valid_points(flat)
        if len(valid) > 0:
            return float(valid[0, 0]), float(valid[0, 1])
    return None

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
                        alpha=track_alpha, zorder=2,linestyle="--")
            #ax.plot(points[:, 0], points[:, 1], color=track.color,linestyle="--",linewidth=3, alpha=track_alpha, zorder=4)

    # --- Floaters: dashed trajectory + label on the first known position ---
    if xy_floaters.size > 0:
        for floater_index in range(xy_floaters.shape[1]):
            trajectory = valid_points(xy_floaters[:, floater_index, :])
            if len(trajectory) == 0:
                continue

            if len(trajectory) > 1:
                ax.plot(trajectory[:, 0], trajectory[:, 1], color=FLOATER_COLOR,
                        linewidth=2, alpha=0.5, zorder=4)

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
        ax.plot([], [], marker="o", color=FLOATER_COLOR, linestyle="None", label="Floaters")
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

    plt.tight_layout()
    plt.savefig(output_file, dpi=900, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Local map saved in: {output_file}")

    return output_file