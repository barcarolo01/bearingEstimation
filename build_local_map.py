import numpy as np
import matplotlib.pyplot as plt

FONTSIZE = 18

def _is_valid(*values):
    """Restituisce True solo se nessuno dei valori è NaN o None."""
    return all(v is not None and not np.isnan(float(v)) for v in values)

def build_local_cartesian_map(
    floaters_coordinates, 
    TX_positions_coordinates, 
    estimated_vessel_coordinates, 
    center_coordinates, 
    window_width_m, 
    window_height_m, 
    output_file="map_local.png", 
    track_TX=False, 
    track_estimated=False,
    track_floaters=True,
):
    """
    Genera e salva un grafico cartesiano locale in metri basato su un centro custom.
    
    Parametri
    ---------
    floaters_coordinates : array-like di forma (N, M, 3) oppure None
        N = numero di posizioni temporali, M = numero di floater, 3 = [lat, lon, depth].
        Se fornito un array (M, 3) o (M, 2), viene trattato come singolo istante temporale (N=1).
    TX_positions_coordinates, estimated_vessel_coordinates : array-like di forma (N, 2/3) oppure None
    center_coordinates : array-like di forma (2,) -> [lat_centro, lon_centro]
    window_width_m : float -> Larghezza della finestra di visualizzazione in metri
    window_height_m : float -> Altezza della finestra di visualizzazione in metri
    track_floaters : bool -> se True, disegna la traiettoria tratteggiata di ogni floater
    """
    
    # Raggio della Terra in metri
    R = 6371000.0
    lat_ref = np.radians(center_coordinates[0])
    lon_ref = np.radians(center_coordinates[1])

    def _geo_to_local(arr):
        """
        Converte coordinate [Lat, Lon, (Depth)] in [X, Y, (Depth)] in metri rispetto al centro.
        Supporta array di forma arbitraria (..., 2) o (..., 3): l'ultima dimensione
        deve contenere [lat, lon, (depth)], tutte le dimensioni precedenti vengono preservate.
        """
        if arr is None:
            return np.empty((0, 2))

        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return np.empty((0,) + arr.shape[1:])

        orig_shape = arr.shape
        flat = arr.reshape(-1, orig_shape[-1])

        lats = np.radians(flat[:, 0])
        lons = np.radians(flat[:, 1])

        # Proiezione locale (Equirettangolare)
        x = R * (lons - lon_ref) * np.cos(lat_ref)
        y = R * (lats - lat_ref)

        if orig_shape[-1] >= 3:
            depths = flat[:, 2]
            depths = np.where(depths == -999, np.nan, depths)
            out = np.column_stack((x, y, depths))
        else:
            out = np.column_stack((x, y))

        return out.reshape(orig_shape[:-1] + (out.shape[-1],))

    def _depth_str(depth):
        return f"{depth:.1f}m" if not np.isnan(depth) else "N/A"

    # --- Normalizzazione floaters a forma (N, M, C) ---
    xy_fl = _geo_to_local(floaters_coordinates)
    if xy_fl.size > 0 and xy_fl.ndim == 2:
        # Input (M, C) -> singolo istante temporale, aggiunge asse N=1
        xy_fl = xy_fl[np.newaxis, :, :]

    xy_tx = _geo_to_local(TX_positions_coordinates)
    xy_est = _geo_to_local(estimated_vessel_coordinates)

    # Filtraggio dei punti validi (rimozione NaN)
    tx_valid = xy_tx[[_is_valid(pt[0], pt[1]) for pt in xy_tx]] if len(xy_tx) > 0 else []
    est_valid = xy_est[[_is_valid(pt[0], pt[1]) for pt in xy_est]] if len(xy_est) > 0 else []

    # Inizializzazione Grafico Matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, linestyle='--', alpha=0.6, color='#cccccc')
    

    # --- Traiettoria TX ---
    if TX_positions_coordinates is not None and track_TX and len(tx_valid) > 1:
        ax.plot(tx_valid[:, 0], tx_valid[:, 1], color="#FFD700", linewidth=3, alpha=0.7, zorder=1)

    # --- Punti TX (Gialli) ---
    if TX_positions_coordinates is not None:
        for i, pt in enumerate(tx_valid):
            depth_val = pt[2] if pt.shape[0] >= 3 else np.nan
            ax.plot(pt[0], pt[1], marker='o', color="#FFD700", markersize=10, markeredgecolor='black', zorder=3)
            #ax.text(pt[0]+2, pt[1]+2, f"TX{i+1}\n({_depth_str(depth_val)})", fontsize=8, color='#555500')

    # --- Traiettoria Stimata ---
    if estimated_vessel_coordinates is not None and track_estimated and len(est_valid) > 1:
        ax.plot(est_valid[:, 0], est_valid[:, 1], color="#00CC66", linewidth=3, linestyle='--', alpha=0.7, zorder=2)

    # --- Punti Stimati (Verdi) ---
    if estimated_vessel_coordinates is not None:
        for i, pt in enumerate(est_valid):
            depth_val = pt[2] if pt.shape[0] >= 3 else np.nan
            ax.plot(pt[0], pt[1], marker='o', color="#00CC66", markersize=8, markeredgecolor='black', zorder=4)
            #ax.text(pt[0]+2, pt[1]-4, f"Est{i+1}\n({_depth_str(depth_val)})", fontsize=8, color='#005522')

    # --- Floaters: traiettoria tratteggiata per ciascun floater + etichetta sull'ultima posizione ---
    if floaters_coordinates is not None and xy_fl.size > 0:
        n_positions, n_floaters = xy_fl.shape[0], xy_fl.shape[1]

        for m in range(n_floaters):
            floater_traj = xy_fl[:, m, :]  # (N, 2 o 3)
            valid_mask = [_is_valid(pt[0], pt[1]) for pt in floater_traj]
            valid_traj = floater_traj[valid_mask]

            if len(valid_traj) == 0:
                continue

            # --- Traiettoria tratteggiata (posizioni consecutive) ---
            # --- Traiettoria tratteggiata (posizioni consecutive) ---
            if track_floaters and len(valid_traj) > 1:
                ax.plot(
                    valid_traj[:, 0], valid_traj[:, 1],
                    color='#FF0000', linewidth=1.5, linestyle='--', alpha=0.6, zorder=4
                )
                # Marker su tutte le posizioni tranne la prima (che ha già l'etichetta)
                ax.plot(
                    valid_traj[1:, 0], valid_traj[1:, 1],
                    marker='o', markersize=4, color='#FF0000', linestyle='None', alpha=0.6, zorder=4
                )

            # --- Etichetta sulla prima posizione nota ---
            first_pt = valid_traj[0]
            label = f"F{m+1}"
            ax.text(
                first_pt[0], first_pt[1], label, 
                color='white',
                weight='bold', 
                fontsize=10, 
                fontfamily='monospace',
                va='center',
                ha='center',
                zorder=5,
                bbox=dict(
                    facecolor='#FF0000', 
                    edgecolor='black', 
                    boxstyle='circle,pad=0.2',
                    linewidth=1.5,
                )
            )
        

    # --- Dimensionamento Finestra in Metri ---
    half_w = window_width_m / 2.0
    half_h = window_height_m / 2.0
    
    ax.set_xlim(-half_w, half_w)
    ax.set_ylim(-half_h, half_h)

    # Etichette assi e dettagli grafici
    ax.set_xlabel("West-East [meters]", fontsize=FONTSIZE, fontweight='bold')
    ax.set_ylabel("South-North [meters]", fontsize=FONTSIZE, fontweight='bold')
    
    ax.set_aspect('equal', adjustable='box')
    
    # Fake plot per la legenda ordinata
    label_centro = f"Center coordinates\n[{center_coordinates[0]:.5f}, {center_coordinates[1]:.5f}]"
    ax.plot(0, 0, 'kx', markersize=5, markeredgewidth=2, label=label_centro)
    
    if floaters_coordinates is not None:
        ax.plot([], [], marker='s', color='#FF0000', linestyle='None', label='Floaters')
    if TX_positions_coordinates is not None:
        ax.plot([], [], marker='o', color='#FFD700', linestyle='None', label='Groung truth')
    if estimated_vessel_coordinates is not None:
        ax.plot([], [], marker='o', color='#00CC66', linestyle='None', label='Estimated positions')
    
    #ax.legend(loc="upper right", frameon=True, facecolor='white', edgecolor='grey', fontsize=FONTSIZE)

    # Salvataggio ed output
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mappa locale salvata in: {output_file}")

# --- ESEMPIO DI UTILIZZO ---
if __name__ == '__main__':
    # Coordinate di esempio: [Lat, Lon, Profondità]
    RX_Coords = np.array([
        [44.4056, 8.9463, -10],
        [44.4070, 8.9490, -15]
    ])
    TX_Coords = np.array([
        [44.4060, 8.9470, -20],
        [44.4065, 8.9480, -22]
    ])
    Est_Coords = np.array([
        [44.4058, 8.9472, -999],
        [44.4063, 8.9478, -12]
    ])
    
    RX_Coords = np.load("Synth/RX_Coordinates.npy")
    TX_Coords = np.load("Synth/TX_Coordinates.npy")
    Est_Coords = np.load("Synth/Estimated_Coordinates.npy")
    center = np.load("Synth/Center_Coordinates.npy")
    

    

    # Finestra di 1000 metri di larghezza e 800 metri di altezza
    print(RX_Coords)
    build_local_cartesian_map(
        RX_Coords,
        TX_Coords, 
        Est_Coords, 
        center_coordinates=RX_Coords[0,:2],
        window_width_m=40, 
        window_height_m=40,
        output_file="map_local.png",
        track_TX=True,
        track_estimated=True
    )