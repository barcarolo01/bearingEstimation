"""
Shared utilities for `build_local_map.py` (matplotlib, local cartesian plane)
and `build_folium_map.py` (folium, web map).

Both drawing functions accept the same parameters:

    floater_coordinates            (N, M, 2|3) or (M, 2|3)
    TX_coordinates                 (K, 2|3)
    estimated_vessel_coordinates   (K, 2|3)
    tracks                         list of Track (see below)
    output_file, track_alpha

The local map additionally takes `center_coordinates`, `window_width_m` and
`window_height_m`, which are meaningless on folium.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

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
