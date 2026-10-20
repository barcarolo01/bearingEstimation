import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, RadioButtons, Button
from matplotlib.figure import Figure
from digitalshadow.Positioning.kalman_filter import *
from digitalshadow.Positioning.coordinate_generator import *
from digitalshadow.Positioning.tracking_plots import *

EARTH_RADIUS = 6378137.0
KALMAN_DT = 1.0              # time between two consecutive steps [s]

# --- Font sizes ---------------------------------------------------------------
FONTSIZE = 16                # everything shown on screen (titles, axes, ticks, widgets, annotations)
FONTSIZE_LEGEND = 14         # legends only
SAVE_DPI = 900               # resolution of the PNGs saved by the "Save figures" button
EXPORT_MAP_SIZE = (8, 8)     # size [inches] of the exported map figure
EXPORT_FERR_SIZE = (8, 5)    # size [inches] of the exported floater-error figure

plt.rcParams.update({
    "font.size":         FONTSIZE,
    "axes.titlesize":    FONTSIZE,
    "axes.labelsize":    FONTSIZE,
    "xtick.labelsize":   FONTSIZE,
    "ytick.labelsize":   FONTSIZE,
    "figure.titlesize":  FONTSIZE,
    "legend.fontsize":   FONTSIZE_LEGEND,
    "legend.title_fontsize": FONTSIZE_LEGEND,
})

FLOATER_STYLE = {
    "Floater GT":      dict(color="#FF0000", lw=2.5),
    "IMU":             dict(color="#0000FF", lw=2),
    "IMU+MDS":         dict(color="#2AB040", lw=2),
    "IMU compensated": dict(color="#D6940F", lw=2),
}

TARGET_STYLE = {
    "Target GT":     dict(color="#000000", lw=3, ls="--"),
    "Triangulation": dict(color="#002AFF", zorder=99, ls="none", marker="x", ms=3, alpha=0.75),
    "Kalman":        dict(color="#FFCC00FF", lw=3),
    "Kalman RTS": dict(color="#2AB040", lw=3),
}

# Floater positions used as input for the triangulation: label -> key in results.npz
KALMAN_SOURCES = {
    "Floater GT":      "RX",
    "IMU":             "RX_IMU",
    "IMU+MDS":         "RX_IMU_MDS",
    "IMU compensated": "RX_IMU_comp",

}

def load_results(simulation_folder):
    # Load the numpy array with the simulation results
    res = np.load(os.path.join(simulation_folder, "results.npz"))
    center = np.asarray(res["Center_coordinates"]).ravel()
    lat0, lon0 = (center[0], center[1])

    floater_positions = {
        "IMU": res["RX_IMU"],
        "IMU+MDS": res["RX_IMU_MDS"],
        "IMU compensated": res["RX_IMU_comp"],
    }
    gt = res["RX"]

    # Horizontal error at each step: shape (NUMBER_OF_FLOATERS, SIMULATION_STEPS)
    errors = {k: np.linalg.norm(v[:, :, :2] - gt[:, :, :2], axis=-1).T for k, v in floater_positions.items()}
    return res, lat0, lon0, gt, floater_positions, errors


def extract_kalman_tracks(kres):
    """Return the target floater_positions from the output of estimate_TX_position():
      "Triangulation": raw find_points fixes (no filtering)
      "Kalman":        causal Kalman filter
      "Kalman RTS":    Kalman + RTS smoother
    Each array has shape (N_STEPS, 3) [x, y, depth] in local metres (same frame as RX/TX);
    NaN steps (failed fixes, steps before the filter starts) are simply not drawn.
    If the filter could not start, estimate_TX_position() returns None for both Kalman tracks."""
    return {"Triangulation": kres.get("raw"),
            "Kalman": kres.get("filtered"),
            "Kalman RTS": kres.get("smoothed")}


def target_rmse(arr, tx_gt):
    """Horizontal RMSE of a target track vs ground truth, ignoring NaN steps (None -> NaN)."""
    if arr is None:
        return np.nan
    e = np.linalg.norm(arr[:, :2] - tx_gt[:, :2], axis=-1)
    return np.sqrt(np.nanmean(e ** 2)) if np.isfinite(e).any() else np.nan


def run_all_kalman(res):
    """Run triangulation + Kalman once per floater-position source in KALMAN_SOURCES.
    Prints a tracking report for each case and a final comparison table.
    Returns {source label: estimate_TX_position output}."""
    tx_gt = np.asarray(res["TX"])
    kres_all = {}
    for label, key in KALMAN_SOURCES.items():
        print("\n" + "=" * 60)
        print(f"Triangulation + Kalman using floater positions: {label}  ({key})")
        print("=" * 60)
        kres = estimate_TX_position(res[key], res["bearings_no_psi_error"], res["bearing_weights"]**2, dt=KALMAN_DT,
                            elevation_array=res["elevations"])
        try:
            tracking_report(kres, res["TX"])
        except Exception as exc:   # e.g. the filter could not start (filtered/smoothed = None)
            print(f"tracking_report failed: {exc}")
        kres_all[label] = kres

    # Summary: target RMSE for every source / estimator
    names = ["Triangulation", "Kalman", "Kalman RTS"]
    print("\n" + "=" * 60)
    print("Target RMSE [m] by floater-position source")
    print(f"{'Floater positions':22s}" + "".join(f"{n:>15s}" for n in names))
    for label, kres in kres_all.items():
        tracks = extract_kalman_tracks(kres)
        print(f"{label:22s}" + "".join(f"{target_rmse(tracks[n], tx_gt):15.3f}" for n in names))
    print("=" * 60 + "\n")
    return kres_all



def plot_simulation_results(simulation_folder):
    res, lat0, lon0, gt, floater_positions, errors = load_results(simulation_folder)
    n_floaters = int(res["NUMBER_OF_FLOATERS"])

    # Triangulation + Kalman filter, once per floater-position source (with reports)
    kres_all = run_all_kalman(res)
    """kres_all: {source label: estimate_TX_position output}, as returned by run_all_kalman()."""
    res, lat0, lon0, gt, floater_positions, errors = load_results(simulation_folder)
    n_floaters = int(res["NUMBER_OF_FLOATERS"])
    tx_gt = np.asarray(res["TX"])            # single transmitter, shape (N_STEPS, 3)
    kres_all = kres_all or {}

    # Target floater_positions for each floater-position source
    tracks_by_source = {label: extract_kalman_tracks(k) for label, k in kres_all.items()}
    for label, tracks in tracks_by_source.items():
        for name, arr in tracks.items():
            if arr is None:
                print(f"Warning: '{name}' track not available with {label} positions, it will not be drawn.")

    fig = plt.figure(figsize=(15, 7.5))
    ax_map = fig.add_axes([0.25, 0.09, 0.42, 0.83])
    ax_ferr = fig.add_axes([0.73, 0.09, 0.25, 0.50])   # bottom right: floater error

    # --- Widgets -----------------------------------------------------------
    # Top right: which floater positions were used for triangulation + Kalman.
    # Each label also shows the Kalman RTS target RMSE obtained with that source.
    source_labels = {}
    for label in tracks_by_source:
        rmse = target_rmse(tracks_by_source[label]["Kalman RTS"], tx_gt)
        info = f"RTS RMSE {rmse:.2f} m" if np.isfinite(rmse) else "Kalman not started"
        source_labels[f"{label}"] = label
    ax_source = fig.add_axes([0.73, 0.70, 0.25, 0.22])
    ax_source.set_title("Floater positions for triangulation + Kalman", fontsize=FONTSIZE, loc="left")
    radio_source = RadioButtons(ax_source, list(source_labels) or ["(not available)"])
    ax_coord = fig.add_axes([0.02, 0.83, 0.17, 0.10])
    ax_coord.set_title("Coordinates", fontsize=FONTSIZE, loc="left")
    radio_coord = RadioButtons(ax_coord, ["Local (m)", "Geographic (°)"])

    floater_traces = list(FLOATER_STYLE)
    ax_check_f = fig.add_axes([0.02, 0.62, 0.17, 0.17])
    ax_check_f.set_title("Floaters", fontsize=FONTSIZE, loc="left")
    check_f = CheckButtons(ax_check_f, floater_traces, [True] * len(floater_traces))

    target_traces = list(TARGET_STYLE)
    ax_check_t = fig.add_axes([0.02, 0.43, 0.17, 0.15])
    ax_check_t.set_title("Target", fontsize=FONTSIZE, loc="left")
    check_t = CheckButtons(ax_check_t, target_traces, [True] * len(target_traces))

    select_labels = ["All"] + [f"Floater {i + 1}" for i in range(n_floaters)]
    h = min(0.27, 0.035 * len(select_labels))
    ax_select = fig.add_axes([0.02, 0.38 - h, 0.17, h])
    ax_select.set_title("Show floater", fontsize=FONTSIZE, loc="left")
    radio_select = RadioButtons(ax_select, select_labels)

    # Bottom left: regenerate the two plots as separate figures and save them
    ax_save_figs = fig.add_axes([0.02, 0.03, 0.17, 0.05])
    btn_save_figs = Button(ax_save_figs, "Save figures")

    # Make sure every widget label uses FONTSIZE
    for w in (radio_source, radio_coord, check_f, check_t, radio_select):
        for t in w.labels:
            t.set_fontsize(FONTSIZE)
    btn_save_figs.label.set_fontsize(FONTSIZE)

    state = {"geo": False, "floater": None,     # floater: 0-based index or None (= all)
                "source": next(iter(tracks_by_source), None)}

    def get_map_coordinates(arr):
        # Concert geographical coordinates to local frame
        if state["geo"]:
            geo = local_to_geo(np.asarray([lat0,lon0]), arr[..., :2])
            return geo[..., 1], geo[..., 0]   # (lon, lat)
    
        return arr[..., 0], arr[..., 1]
    
    def current_view():
        visible = dict(zip(floater_traces, check_f.get_status()))
        visible.update(zip(target_traces, check_t.get_status()))
        floaters = range(n_floaters) if state["floater"] is None else [state["floater"]]
        return visible, floaters

    def draw_map(ax, visible, floaters, with_title=True):
        # Target tracks: ground truth + floater_positions obtained with the selected source
        targets = {"Target GT": tx_gt}
        targets.update(tracks_by_source.get(state["source"],
                                            {"Triangulation": None, "Kalman": None, "Kalman RTS": None}))

        # --- Floaters ---------------------------------------------------------
        floater_arrays = {"Floater GT": gt, **floater_positions}
        for name, arr in floater_arrays.items():
            if not visible[name]:
                continue
            for i, n in enumerate(floaters):
                x, y = get_map_coordinates(arr[:, n, :])
                ax.plot(x, y, **FLOATER_STYLE[name], label=name if i == 0 else None)
                if name == "Floater GT":
                    ax.plot(x[0], y[0], "o", color="#FF0000", ms=5, zorder=999)

        # Floater number (1-based) next to the start of each trajectory
        if any(visible[name] for name in floater_traces):
            for n in floaters:
                x, y = get_map_coordinates(gt[0, n, :])
                ax.annotate(str(n + 1), (x, y), xytext=(-8, 6), textcoords="offset points",
                            ha="right", fontsize=FONTSIZE, fontweight="bold", zorder=6,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="k", lw=0.8))

        # --- Target -------------------------------------------------------------
        for name, arr in targets.items():
            if not visible[name] or arr is None:
                continue
            x, y = get_map_coordinates(arr)
            ax.plot(x, y, **TARGET_STYLE[name], label=name)
            if name == "Target GT":
                c = TARGET_STYLE[name]["color"]
                ax.plot(x[0], y[0], "o", color=c, ms=8, mfc="white", mew=2,
                        label="Target start", zorder=5)
                ax.plot(x[-1], y[-1], "o", color=c, ms=8, label="Target end", zorder=5)

        if state["geo"]:
            ax.set_xlabel("Longitude [°]")
            ax.set_ylabel("Latitude [°]")
            ax.set_aspect(1 / np.cos(np.radians(lat0)), adjustable="datalim")
            ax.ticklabel_format(useOffset=False, style="plain")
        else:
            ax.set_xlabel("East [m]")
            ax.set_ylabel("North [m]")
            ax.set_aspect("equal", adjustable="datalim")
        if with_title:
            title = "Trajectories" + ("" if state["floater"] is None else f" – Floater {state['floater'] + 1}")
            if state["source"] is not None:
                title += f"\nTarget estimated with {state['source']} floater positions"
            ax.set_title(title)
        ax.grid(True, ls="--", alpha=.5)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=FONTSIZE_LEGEND)

    def draw_ferr(ax, visible, floaters, with_title=True):
        for name in floater_positions:
            if not visible[name]:
                continue
            e = errors[name][list(floaters)]            # (n_selected, steps)
            rmse = np.sqrt((e ** 2).mean())
            if len(e) == 1:
                ax.plot(e[0], **FLOATER_STYLE[name], label=f"{name} (RMSE {rmse:.2f} m)")
            else:
                # Mean across floaters + interquartile range (25th-75th percentile) as a band
                steps = np.arange(e.shape[1])
                q1, q3 = np.percentile(e, [25, 75], axis=0)
                ax.fill_between(steps, q1, q3, color=FLOATER_STYLE[name]["color"],
                                alpha=0.2, lw=0)
                ax.plot(steps, e.mean(axis=0), **FLOATER_STYLE[name],
                        label=f"{name} (RMSE {rmse:.2f} m)")
        if len(floaters) > 1 and any(visible[name] for name in floater_positions):
            ax.fill_between([], [], [], color="grey", alpha=0.2, lw=0)
        if with_title:
            ax.set_title("Floater error vs ground truth")
        ax.set_xlabel("Simulation steps")
        ax.set_ylabel("Meters")
        ax.grid(True, ls="--", alpha=.5)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", fontsize=FONTSIZE_LEGEND)

    def redraw(_=None):
        visible, floaters = current_view()
        ax_map.clear()
        ax_ferr.clear()
        draw_map(ax_map, visible, floaters)
        draw_ferr(ax_ferr, visible, floaters)
        fig.canvas.draw_idle()
        fig.savefig(os.path.join(simulation_folder,"sim_results.png"))

    def current_tag():
        tag = "all" if state["floater"] is None else f"floater_{state['floater'] + 1}"
        tag += "_geo" if state["geo"] else "_local"
        if state["source"] is not None:
            tag += "_kf-" + state["source"].replace(" ", "_").replace("+", "_")
        return tag

    def on_coord(label):
        state["geo"] = label.startswith("Geo")
        redraw()

    def on_source(label):
        if label in source_labels:
            state["source"] = source_labels[label]
            redraw()

    def on_select(label):
        state["floater"] = None if label == "All" else int(label.split()[-1]) - 1
        redraw()

    def on_save_figs(_):
        """Regenerate from scratch the map and the floater-error plot as two separate
        figures (same content as displayed, without titles), save them as PNG (SAVE_DPI)
        and discard them without showing."""
        visible, floaters = current_view()
        tag = current_tag()
        plots = {"map": (draw_map, EXPORT_MAP_SIZE),
                 "floater_error": (draw_ferr, EXPORT_FERR_SIZE)}
        for name, (draw, size) in plots.items():
            f = Figure(figsize=size)          # standalone figure, not attached to pyplot -> never shown
            ax = f.add_subplot(111)
            draw(ax, visible, floaters, with_title=False)
            f.tight_layout()
            path = os.path.join(simulation_folder, f"{name}_{tag}.png")
            f.savefig(path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.05)
            print(f"Saved: {path}")

    radio_source.on_clicked(on_source)
    radio_coord.on_clicked(on_coord)
    check_f.on_clicked(redraw)
    check_t.on_clicked(redraw)
    radio_select.on_clicked(on_select)
    btn_save_figs.on_clicked(on_save_figs)

    # Keep references to the widgets (otherwise garbage collection disables them)
    fig._widgets = (radio_coord, check_f, check_t, radio_select, btn_save_figs, radio_source)
    redraw()
    plt.show()


if __name__ == "__main__":
        plot_simulation_results("simulationCIRCLE")