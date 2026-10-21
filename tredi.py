from PIL import Image
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import os
from src.plot_simulation import estimate_TX_position, load_results

FONTSIZE = 18
R_EARTH = 6371000.0  # Raggio terrestre [m]
 
# Stili delle tracce
STYLES = {
    "Target GT":     dict(color="#000000", lw=3, ls="--"),
    "Triangulation": dict(color="#000DFF", zorder=99, ls="--", marker="o", ms=3, alpha=0.5),
    "Floater GT":    dict(color="#FF0000", lw=2.5),
}
 
 
def build_local_cartesian_map_3d(
    floaters_coordinates,
    tx_true_coordinates,
    tx_estimated_coordinates,
    center_coordinates,
    window_width_m=None,       # se None viene calcolata automaticamente dai dati
    window_height_m=None,      # se None viene calcolata automaticamente dai dati
    max_depth_m=100.0,         # limite inferiore dell'asse Z
    margin_m=50.0,             # margine attorno ai dati per la finestra automatica
    label_floaters=True,       # numero del floater sulla prima posizione
    crop_px=(800, 500, 200),   # (sopra, sotto, destra) pixel da tagliare; None = nessun ritaglio
    out_path="local_map_3D.png",
    show=True,
):
    """
    Mappa 3D locale con assi etichettati in coordinate geografiche.
 
    Parametri
    ---------
    floaters_coordinates : array (N, M, 2/3) oppure (M, 2/3), o None
        N = istanti temporali, M = numero di floater.
    tx_true_coordinates : array (K, 2/3), o None
        Posizioni reali del trasmettitore (ground truth).
    tx_estimated_coordinates : array (K, 2/3), o None
        Posizioni stimate del trasmettitore (triangolazione).
    center_coordinates : array-like [lat, lon]
        Centro della mappa in gradi: origine del riferimento cartesiano locale.
 
    Floater, ground truth e stime sono già in coordinate cartesiane locali (metri)
    con origine nel centro: ultima dimensione [x (Est), y (Nord), (depth)],
    profondità positiva verso il basso (-999 = valore mancante).
    Le etichette degli assi X/Y vengono convertite in longitudine/latitudine
    a partire dal centro.
    """
    lat0_deg = float(center_coordinates[0])
    lon0_deg = float(center_coordinates[1])
 
    m_per_deg_lat = R_EARTH * np.pi / 180.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.radians(lat0_deg))
 
    def _to_local(arr):
        """
        Normalizza un array cartesiano locale [x, y, (depth)] in metri a [x, y, z].
        Preserva le dimensioni iniziali; se manca la profondità assume z = 0.
        """
        if arr is None:
            return np.empty((0, 3))
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return np.empty((0, 3))
 
        orig_shape = arr.shape
        flat = arr.reshape(-1, orig_shape[-1])
        x = flat[:, 0]
        y = flat[:, 1]
 
        if orig_shape[-1] >= 3:
            depths = np.where(flat[:, 2] == -999, np.nan, flat[:, 2])
        else:
            depths = np.zeros_like(x)  # nessuna profondità -> superficie
 
        out = np.column_stack((x, y, depths))
        return out.reshape(orig_shape[:-1] + (3,))
 
    def _valid(points):
        """Mantiene solo i punti con X, Y, Z finiti."""
        if points.size == 0:
            return np.empty((0, 3))
        return points[np.isfinite(points).all(axis=-1)]
 
    # --- Conversione in coordinate locali ---
    xyz_fl = _to_local(floaters_coordinates)
    if xyz_fl.size > 0 and xyz_fl.ndim == 2:
        xyz_fl = xyz_fl[np.newaxis, :, :]  # (M, 3) -> (1, M, 3)
 
    tx_true = _valid(_to_local(tx_true_coordinates))
    tx_est = _valid(_to_local(tx_estimated_coordinates))
 
    # --- Figura ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#ffffff")
 
    # Nota: la profondità è positiva verso il basso, quindi si plotta -z.
 
    # --- Floater GT ---
    all_xy = []
    if xyz_fl.size > 0:
        legend_done = False
        for mi in range(xyz_fl.shape[1]):
            traj = _valid(xyz_fl[:, mi, :])
            if len(traj) == 0:
                continue
            all_xy.append(traj[:, :2])
 
            ax.plot(
                traj[:, 0], traj[:, 1], -traj[:, 2],
                label=None if legend_done else "Floater GT",
                **STYLES["Floater GT"],
            )
            legend_done = True
 
            if label_floaters:
                p = traj[0]
                ax.text(
                    p[0], p[1], -p[2], f"{mi + 1}",
                    color="white", weight="bold", fontsize=8, fontfamily="monospace",
                    ha="center", va="center", zorder=1000,
                    bbox=dict(facecolor=STYLES["Floater GT"]["color"], edgecolor="black",
                              boxstyle="circle,pad=0.2", linewidth=1.5),
                )
 
    # --- Target GT (posizioni reali del trasmettitore) ---
    if len(tx_true) > 0:
        all_xy.append(tx_true[:, :2])
        ax.plot(tx_true[:, 0], tx_true[:, 1], -tx_true[:, 2],
                label="Target GT", **STYLES["Target GT"])
 
    # --- Triangulation (posizioni stimate) ---
    if len(tx_est) > 0:
        all_xy.append(tx_est[:, :2])
        ax.plot(tx_est[:, 0], tx_est[:, 1], -tx_est[:, 2],
                label="Triangulation", **STYLES["Triangulation"])
 
    # --- Centro ---
    ax.scatter(0, 0, 0, color="black", marker="x", s=40,
               label=f"Center\n[{lat0_deg:.5f}, {lon0_deg:.5f}]")
 
    # --- Limiti finestra (centrata sul centro) ---
    if all_xy:
        pts = np.vstack(all_xy)
        auto_half_w = np.max(np.abs(pts[:, 0])) + margin_m
        auto_half_h = np.max(np.abs(pts[:, 1])) + margin_m
    else:
        auto_half_w = auto_half_h = 250.0
 
    half_w = window_width_m / 2.0 if window_width_m is not None else auto_half_w
    half_h = window_height_m / 2.0 if window_height_m is not None else auto_half_h
 
    ax.set_xlim(-half_w, half_w)
    ax.set_ylim(-half_h, half_h)
    ax.set_zlim(-max_depth_m, 5)
 
    # --- Tick in coordinate geografiche ---
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune="lower"))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune="lower"))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{lon0_deg + x / m_per_deg_lon:.4f}°"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f"{lat0_deg + y / m_per_deg_lat:.4f}°"))
    # Profondità mostrata come valore positivo
    ax.zaxis.set_major_formatter(ticker.FuncFormatter(lambda z, _: f"{abs(z):.0f}"))
 
    ax.tick_params(axis="x", labelsize=FONTSIZE - 8)
    ax.tick_params(axis="y", labelsize=FONTSIZE - 8)
    ax.tick_params(axis="z", labelsize=FONTSIZE - 6)
 
    ax.set_xlabel("Longitude [°]", fontsize=FONTSIZE, fontweight="bold", labelpad=18)
    ax.set_ylabel("Latitude [°]", fontsize=FONTSIZE, fontweight="bold", labelpad=18)
    ax.set_zlabel("Depth [m]", fontsize=FONTSIZE, fontweight="bold", labelpad=10)
 
    ax.legend(loc="upper right", frameon=True, facecolor="white",
              edgecolor="grey", fontsize=FONTSIZE - 3)
 
    ax.view_init(elev=15, azim=230, roll=0)
 
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.5, dpi=600)
    if show:
        plt.show()
    plt.close(fig)
 
    # --- Ritaglio opzionale dell'immagine ---
    if crop_px is not None:
        top, bottom, right = crop_px
        img = Image.open(out_path)
        w, h = img.size
        if top + bottom < h and right < w:
            img.crop((0, top, w - right, h - bottom)).save(out_path)
 
 
def filtro_mediano(x, W):
    from numpy.lib.stride_tricks import sliding_window_view
    x = np.asarray(x, dtype=float)
    h = W // 2
    y = x.copy()
    y[h:len(x) - h] = np.median(sliding_window_view(x, W), axis=1)
    return y

if __name__ == "__main__":
    res,lat0, lon0, gt, floater_positions, errors = load_results("simulationCIRCLE_3depth_3")
    kres = estimate_TX_position(res["RX"], res["bearings_no_psi_error"], res["bearing_weights"]**2, dt=1.0,elevation_array=res["elevations"])


    estimated = kres["smoothed"]
    estimated_depth = estimated[:,2]
    estimated_depth = filtro_mediano(estimated_depth,5)

    mask = (res["TX"][:,2] != np.nan)
    e = (res["TX"][:,2][mask]-estimated_depth[mask])
    print(f"MEDIAN = {np.median(e)}")
    print(f"RMSE DEPTH = {np.sqrt(np.mean(e**2))}")

    estimated[:,2] = estimated_depth
    build_local_cartesian_map_3d(
        floaters_coordinates=res["RX"],
        tx_true_coordinates=res["TX"],
        tx_estimated_coordinates=estimated,
        center_coordinates=res["Center_coordinates"],
        window_width_m=500,
        window_height_m=500,
        max_depth_m=100.0,
    )
    print("Generated")