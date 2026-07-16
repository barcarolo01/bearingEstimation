from PIL import Image  
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

# Definizione di una costante FONTSIZE di fallback (visto che era presente nel tuo codice)
FONTSIZE = 18

def _is_valid(*values):
    """Restituisce True solo se nessuno dei valori è NaN o None."""
    return all(v is not None and not np.isnan(float(v)) for v in values)

def build_local_cartesian_map_3d(
    floaters_coordinates, 
    TX_positions_coordinates, 
    estimated_vessel_coordinates, 
    center_coordinates, 
    window_width_m, 
    window_height_m, 
    max_depth_m=100.0, # Limite dell'asse Z per la visualizzazione
    track_TX=False, 
    track_estimated=False
):
    """
    Genera una visualizzazione 3D interattiva cartesiana locale in metri.
    Mantiene gli stessi colori e la logica di proiezione della mappa 2D.
    """
    
    # Raggio della Terra in metri
    R = 6371000.0
    lat_ref = np.radians(center_coordinates[0])
    lon_ref = np.radians(center_coordinates[1])

    def _geo_to_local(arr):
        """Converte coordinate [Lat, Lon, (Depth)] in [X, Y, Z] in metri rispetto al centro."""
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return np.empty((0, 3 if arr.shape[1] >= 3 else 2))
        
        lats = np.radians(arr[:, 0])
        lons = np.radians(arr[:, 1])
        
        # Proiezione locale (Equirettangolare)
        x = R * (lons - lon_ref) * np.cos(lat_ref)
        y = R * (lats - lat_ref)
        
        if arr.shape[1] >= 3:
            depths = arr[:, 2]
            # Sostituisce il valore di fallback -999 con NaN
            depths = np.where(depths == -999, np.nan, depths)
            return np.column_stack((x, y, depths))
        
        # Se la profondità non è presente nell'input, assumiamo sia 0 (superficie)
        return np.column_stack((x, y, np.zeros_like(x)))

    # Conversione in coordinate locali (metri)
    xyz_fl = _geo_to_local(floaters_coordinates)
    xyz_tx = _geo_to_local(TX_positions_coordinates)
    xyz_est = _geo_to_local(estimated_vessel_coordinates)

    # Filtraggio dei punti validi (rimozione NaN su X, Y e Z)
    tx_valid = xyz_tx[[_is_valid(pt[0], pt[1], pt[2]) for pt in xyz_tx]] if len(xyz_tx) > 0 else np.empty((0, 3))
    est_valid = xyz_est[[_is_valid(pt[0], pt[1], pt[2]) for pt in xyz_est]] if len(xyz_est) > 0 else np.empty((0, 3))

    # Inizializzazione Grafico Matplotlib 3D
    fig = plt.figure(figsize=(12, 10))
    plt.tight_layout()
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')

    # --- Traiettoria TX (Gialla) ---
    if track_TX and len(tx_valid) > 1:
        # Invertiamo Z (-pt[2]) per fare in modo che la profondità vada verso il basso nel plot grafico
        ax.plot(tx_valid[:, 0], tx_valid[:, 1], -tx_valid[:, 2], color="#FFD700", linewidth=3, alpha=0.7, zorder=1)

    # --- Punti TX / Ground Truth (Gialli) ---
    if len(tx_valid) > 0:
        ax.scatter(tx_valid[:, 0], tx_valid[:, 1], -tx_valid[:, 2], 
                   color="#FFD700", s=60, edgecolors='black', depthshade=False, zorder=3, label='Ground truth')

    # --- Traiettoria Stimata (Verde) ---
    if track_estimated and len(est_valid) > 1:
        ax.plot(est_valid[:, 0], est_valid[:, 1], -est_valid[:, 2], 
                color="#00CC66", linewidth=3, linestyle='--', alpha=0.7, zorder=2)

    # --- Punti Stimati (Verdi) ---
    if len(est_valid) > 0:
        ax.scatter(est_valid[:, 0], est_valid[:, 1], -est_valid[:, 2], 
                   color="#00CC66", s=50, edgecolors='black', depthshade=False, zorder=4, label='Estimated positions')

    # --- Floaters (Rossi) ---
    # Usiamo scatter3D disegnando dei quadrati (marker='s') rossi come nel rettangolo 2D
    ax.scatter(xyz_fl[:, 0], xyz_fl[:, 1], -xyz_fl[:, 2], 
               color='#FF0000', marker='s', s=80, edgecolors='black', depthshade=False, zorder=5, label='Floaters')
    
    # Aggiungiamo le etichette di testo "F1, F2..." accanto ai quadratini rossi
    for i, pt in enumerate(xyz_fl):
        ax.text(pt[0] + 8, pt[1] + 8, -pt[2] - 8, f"F{i+1}", color='black', weight='bold', fontsize=8, zorder=1000)

    # --- Configurazione Limiti Assi Finestra ---
    half_w = window_width_m / 2.0
    half_h = window_height_m / 2.0
    
    ax.set_xlim(-half_w, half_w)
    ax.set_ylim(-half_h, half_h)
    # L'asse Z va da -max_depth (fondo del mare) a 5 (un po' sopra la superficie per vedere i floater)
    ax.set_zlim(-max_depth_m, 5)

    # Etichette assi
    ax.set_xlabel("East-West [meters]", fontsize=FONTSIZE, fontweight='bold', labelpad=10)
    ax.set_ylabel("North-South [meters]", fontsize=FONTSIZE, fontweight='bold', labelpad=10)
    ax.set_zlabel("Depth [meters]", fontsize=FONTSIZE, fontweight='bold', labelpad=10)
    
    # Origin marker (Centro mappa)
    label_centro = f"Center\n[{center_coordinates[0]:.5f}, {center_coordinates[1]:.5f}]"
    ax.scatter(0, 0, 0, color='black', marker='x', s=40, label=label_centro)


    # Legenda ordinata
    #ax.legend(loc="upper right", frameon=True, facecolor='white', edgecolor='grey', fontsize=FONTSIZE-3)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(prune='lower'))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(prune='lower'))

    ax.view_init(elev=15, azim=230, roll=0)
    
    
    
    plt.savefig("map_3D.png",bbox_inches='tight',pad_inches=0.5, dpi=600)
    plt.show()

    
    # 2. Apri l'immagine con Pillow e taglia i pixel dal bordo inferiore
    PIXEL_DA_TAGLIARE_SOPRA = 800
    PIXEL_DA_TAGLIARE_SOTTO = 500  # Sostituisci con il numero esatto di pixel da rimuovere

    img = Image.open("map_3D.png")
    larghezza, altezza = img.size

    # Definiamo la scatola di ritaglio (box): (sinistra, alto, destra, basso)
    # Sottraiamo i pixel desiderati dall'altezza totale per tagliare il fondo
    box_ritaglio = (0, PIXEL_DA_TAGLIARE_SOPRA, larghezza-200, altezza - PIXEL_DA_TAGLIARE_SOTTO)
    img_ritagliata = img.crop(box_ritaglio)

    # 3. Salva il file finale definitivo
    img_ritagliata.save("map_3D.png")


if __name__ == "__main__":
    RX_Coordinates = np.load("Synth/RX_Coordinates.npy")
    TX_Coordinates = np.load("Synth/TX_Coordinates.npy")
    center = np.load("Synth/Center_Coordinates.npy")
    intersection_points = np.load("Synth/Estimated_Coordinates.npy")
    build_local_cartesian_map_3d(
                RX_Coordinates, 
                TX_Coordinates, 
                intersection_points, 
                center, 
                500, 
                500, 
                max_depth_m=100.0, # Limite dell'asse Z per la visualizzazione
                track_TX=True, 
                track_estimated=True)
    print("Generated")