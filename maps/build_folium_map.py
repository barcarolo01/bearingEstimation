import folium
import folium.plugins
import numpy as np
from folium import MacroElement
from jinja2 import Template

class ScaleBar(MacroElement):
    """Barra di scala fissa che si aggiorna automaticamente con lo zoom."""
    
    _template = Template("""
        {% macro script(this, kwargs) %}
        
        // Crea il contenitore della scala
        var scaleDiv = L.control({position: 'bottomleft'});
        
        scaleDiv.onAdd = function(map) {
            var div = L.DomUtil.create('div', 'scale-bar');
            div.style.cssText = `
                background: white;
                border: 2px solid #333;
                border-top: none;
                padding: 2px 5px;
                font-size: 11px;
                font-family: Arial, sans-serif;
                color: #333;
                pointer-events: none;
                min-width: 60px;
                text-align: center;
            `;
            return div;
        };
        
        scaleDiv.addTo({{ this._parent.get_name() }});
        
        function updateScale() {
            var map = {{ this._parent.get_name() }};
            var center = map.getCenter();
            var bounds = map.getBounds();
            
            // Calcola la larghezza della mappa in metri
            var leftPoint = map.latLngToContainerPoint(
                L.latLng(center.lat, bounds.getWest())
            );
            var rightPoint = map.latLngToContainerPoint(
                L.latLng(center.lat, bounds.getEast())
            );
            var pixelWidth = rightPoint.x - leftPoint.x;
            
            // Distanza reale in metri per tutta la larghezza
            var realWidth = center.distanceTo(
                L.latLng(center.lat, bounds.getEast())
            ) * 2;
            
            // Scala: metri per pixel
            var metersPerPixel = realWidth / pixelWidth;
            
            // Larghezza target della barra (100px) → distanza reale
            var targetPixels = 100;
            var targetMeters = metersPerPixel * targetPixels;
            
            // Arrotonda a un numero "bello"
            var magnitude = Math.pow(10, Math.floor(Math.log10(targetMeters)));
            var nice = [1, 2, 5, 10];
            var niceMeters = magnitude;
            for (var i = 0; i < nice.length; i++) {
                if (nice[i] * magnitude >= targetMeters * 0.5) {
                    niceMeters = nice[i] * magnitude;
                    break;
                }
            }
            
            // Pixel effettivi per la distanza arrotondata
            var barPixels = niceMeters / metersPerPixel;
            
            // Etichetta
            var label;
            if (niceMeters >= 1000) {
                label = (niceMeters / 1000) + ' km';
            } else {
                label = niceMeters + ' m';
            }
            
            // Aggiorna DOM
            var div = document.querySelector('.scale-bar');
            if (div) {
                div.style.width = barPixels + 'px';
                div.style.minWidth = 'unset';
                div.innerHTML = label;
            }
        }
        
        // Aggiorna alla creazione e ad ogni zoom/spostamento
        {{ this._parent.get_name() }}.on('zoomend moveend load', updateScale);
        setTimeout(updateScale, 300);
        
        {% endmacro %}
    """)
    
    def __init__(self):
        super().__init__()

def _is_valid(*values):
    """Restituisce True solo se nessuno dei valori è NaN o None."""
    return all(v is not None and not np.isnan(float(v)) for v in values)

def build_map(
    floaters_coordinates, 
    TX_positions_coordinates, 
    estimated_vessel_coordinates, 
    output_file, 
    track_TX=False, 
    track_estimated=False,
    track_floaters=True,
    RX_fw_IMU=None,
    RX_fw_IMU_MDS=None,
    RX_bw_IMU=None,
    RX_bw_IMU_MDS=None,
    imu_alpha=0.4,
):
    """
    Costruisce, salva e restituisce la mappa Folium con:
      - Floaters (F1, F2, ...) come marker con etichetta rossa circolare sulla prima posizione,
        ed eventuale traiettoria tratteggiata che collega le posizioni successive
      - Punti TX in giallo  (NaN ignorati)
      - Punti stimati in verde (NaN ignorati)
      - Traiettorie opzionali che collegano i punti sequenzialmente
      - Traiettorie RX IMU (fw/bw, standard/MDS) sempre disegnate, in forma semi-trasparente

    Per tutti gli array di coordinate, la profondità è opzionale:
    se presente come ultima colonna viene mostrata nel popup, altrimenti "N/A".
    I valori -999 sono trattati come profondità non disponibile.

    Parametri
    ---------
    floaters_coordinates : array-like di forma (N, M, 3) o (N, M, 2), oppure None
        N = numero di posizioni temporali, M = numero di floater.
        Se fornito un array (M, 3) o (M, 2), viene trattato come singolo istante (N=1).
    TX_positions_coordinates : array-like di forma (K, 2) o (K, 3), oppure None
    estimated_vessel_coordinates : array-like di forma (K, 2) o (K, 3), oppure None
    output_file : str
    track_TX : bool, opzionale (default=False)
    track_estimated : bool, opzionale (default=False)
    track_floaters : bool, opzionale (default=True)
        Se True, mostra la traiettoria tratteggiata di ciascun floater.
    RX_fw_IMU, RX_fw_IMU_MDS, RX_bw_IMU, RX_bw_IMU_MDS : array-like di forma (N, M, 3) oppure None
        N = numero di posizioni temporali, M = numero di device, 3 = [lat, lon, depth].
        Stessa convenzione di forma di `floaters_coordinates` (accetta anche (M, 3)/(M, 2)
        come singolo istante temporale, con N=1 aggiunto automaticamente).
        Traiettorie del ricevitore sempre disegnate come linea connessa (nessun flag di attivazione):
        RX_fw_IMU in blu, RX_fw_IMU_MDS in arancione, RX_bw_IMU in verde chiaro, RX_bw_IMU_MDS in rosso.
    imu_alpha : float -> livello di trasparenza (0=invisibile, 1=opaco) delle 4 traiettorie RX IMU. Default 0.4.
    """

    def _extract_coords(arr):
        """
        Restituisce (lats, lons, depths) appiattendo tutte le dimensioni tranne l'ultima.
        Ritorna array vuoti se arr è None.
        """
        if arr is None:
            return np.empty(0), np.empty(0), np.empty(0)

        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return np.empty(0), np.empty(0), np.empty(0)

        flat = arr.reshape(-1, arr.shape[-1])
        lats = flat[:, 0]
        lons = flat[:, 1]
        if flat.shape[1] >= 3:
            depths = flat[:, 2]
            depths = np.where(depths == -999, np.nan, depths)
        else:
            depths = np.full(len(lats), np.nan)
        return lats, lons, depths

    def _depth_str(depth):
        """Formatta la profondità per il popup."""
        return f"{depth:.1f} m" if not np.isnan(depth) else "N/A"

    def _normalize_multi(arr):
        """
        Converte input in array (N, M, C), stessa convenzione usata per floaters_coordinates:
        se l'input ha forma (M, C) viene trattato come singolo istante temporale (N=1).
        Ritorna None se arr è None.
        """
        if arr is None:
            return None
        out = np.asarray(arr, dtype=float)
        if out.ndim == 2:
            out = out[np.newaxis, :, :]
        return out

    # --- Normalizzazione floaters a forma (N, M, C) ---
    floaters_arr = _normalize_multi(floaters_coordinates)

    lats_fl,  lons_fl,  depths_fl  = _extract_coords(floaters_arr)
    lats_tx,  lons_tx,  depths_tx  = _extract_coords(TX_positions_coordinates)
    lats_est, lons_est, depths_est = _extract_coords(estimated_vessel_coordinates)

    # --- Normalizzazione traiettorie RX IMU a forma (N, M, C), come floaters_coordinates ---
    fw_imu_arr = _normalize_multi(RX_fw_IMU)
    fw_imu_mds_arr = _normalize_multi(RX_fw_IMU_MDS)
    bw_imu_arr = _normalize_multi(RX_bw_IMU)
    bw_imu_mds_arr = _normalize_multi(RX_bw_IMU_MDS)

    tx_valid  = [(lat, lon, depth)
                 for lat, lon, depth in zip(lats_tx, lons_tx, depths_tx)
                 if _is_valid(lat, lon)]

    est_valid = [(lat, lon, depth)
                 for lat, lon, depth in zip(lats_est, lons_est, depths_est)
                 if _is_valid(lat, lon)]

    fl_valid  = [(lat, lon, depth)
                 for lat, lon, depth in zip(lats_fl, lons_fl, depths_fl)
                 if _is_valid(lat, lon)]

    # --- Centro mappa: primo punto valido disponibile, in ordine di priorità ---
    center = None
    for candidate_list in (fl_valid, tx_valid, est_valid):
        if len(candidate_list) > 0:
            center = (candidate_list[0][0], candidate_list[0][1])
            break

    if center is None:
        # Fallback: prova con le traiettorie RX IMU se non ci sono altri punti validi
        for imu_arr in (fw_imu_arr, fw_imu_mds_arr, bw_imu_arr, bw_imu_mds_arr):
            if imu_arr is not None and imu_arr.size > 0:
                lats_i, lons_i, _ = _extract_coords(imu_arr)
                for lat, lon in zip(lats_i, lons_i):
                    if _is_valid(lat, lon):
                        center = (lat, lon)
                        break
            if center is not None:
                break

    if center is None:
        raise ValueError("Nessuna coordinata valida fornita per centrare la mappa.")

    # Creazione mappa
    m = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap")

    ScaleBar().add_to(m)

    folium.plugins.MeasureControl(
        position="bottomleft",
        primary_length_unit="meters",
        secondary_length_unit="kilometers",
        primary_area_unit="sqmeters",
        secondary_area_unit="sqkilometers"
    ).add_to(m)

    
    if TX_positions_coordinates is not None and track_TX and len(tx_valid) > 1:
        trajectory_coords = [(lat, lon) for lat, lon, _ in tx_valid]
        folium.PolyLine(
            locations=trajectory_coords,
            color="#FFD700",
            weight=5,
            opacity=0.7,
            #dash_array="5, 10",
            tooltip="Traiettoria Stimata Vessel"
        ).add_to(m)

    # --- Punti TX (gialli) ---
    if TX_positions_coordinates is not None:
        for i, (lat, lon, depth) in enumerate(tx_valid):
            folium.CircleMarker(
                location=(lat, lon),
                radius=10,
                color="#FFD700",
                fill=True,
                fill_color="#FFD700",
                fill_opacity=0.9,
                weight=2,
                popup=folium.Popup(
                    f"<b>TX {i+1}</b><br>Lat: {lat:.6f}<br>Lon: {lon:.6f}<br>Depth: {_depth_str(depth)}",
                    max_width=180
                ),
                tooltip=f"TX {i+1}"
            ).add_to(m)

    # Viene disegnata prima dei punti stimati in modo che i marker rimangano visivamente "sopra" la linea
    if estimated_vessel_coordinates is not None and track_estimated and len(est_valid) > 1:
        trajectory_coords = [(lat, lon) for lat, lon, _ in est_valid]
        folium.PolyLine(
            locations=trajectory_coords,
            color="#00CC66",
            weight=5,
            opacity=0.7,
            dash_array="5, 10",
            tooltip="Traiettoria Stimata Vessel"
        ).add_to(m)

    # --- Punti stimati (verdi) ---
    if estimated_vessel_coordinates is not None:
        for i, (lat, lon, depth) in enumerate(est_valid):
            folium.CircleMarker(
                location=(lat, lon),
                radius=7,
                color="#00CC66",
                fill=True,
                fill_color="#00CC66",
                fill_opacity=0.9,
                weight=2,
                popup=folium.Popup(
                    f"<b>Stimato {i+1}</b><br>Lat: {lat:.6f}<br>Lon: {lon:.6f}<br>Depth: {_depth_str(depth)}",
                    max_width=180
                ),
                tooltip=f"Stimato {i+1}"
            ).add_to(m)

    # --- Traiettorie RX IMU (sempre attive, una traiettoria per ciascun device, semi-trasparenti) ---
    _imu_tracks = [
        (RX_fw_IMU, fw_imu_arr, "#0000FF", "RX fw IMU"),
        (RX_fw_IMU_MDS, fw_imu_mds_arr, "#FFA500", "RX fw IMU MDS"),
        (RX_bw_IMU, bw_imu_arr, "#90EE90", "RX bw IMU"),
        (RX_bw_IMU_MDS, bw_imu_mds_arr, "#115511", "RX bw IMU MDS"),
    ]

    for raw_input, imu_arr, color, label_prefix in _imu_tracks:
        if raw_input is None or imu_arr is None or imu_arr.size == 0:
            continue

        n_devices = imu_arr.shape[1]
        for d in range(n_devices):
            device_lats = imu_arr[:, d, 0]
            device_lons = imu_arr[:, d, 1]
            if imu_arr.shape[-1] >= 3:
                device_depths = np.where(imu_arr[:, d, 2] == -999, np.nan, imu_arr[:, d, 2])
            else:
                device_depths = np.full(imu_arr.shape[0], np.nan)

            device_valid = [(lat, lon, depth)
                             for lat, lon, depth in zip(device_lats, device_lons, device_depths)
                             if _is_valid(lat, lon)]

            if len(device_valid) == 0:
                continue

            # Traiettoria connessa (sempre disegnata di default, semi-trasparente)
            if len(device_valid) > 1:
                trajectory_coords = [(lat, lon) for lat, lon, _ in device_valid]
                folium.PolyLine(
                    locations=trajectory_coords,
                    color=color,
                    weight=3,
                    opacity=imu_alpha,
                    tooltip=f"{label_prefix} - device {d+1}"
                ).add_to(m)

            # Marker sui singoli punti
            for i, (lat, lon, depth) in enumerate(device_valid, start=1):
                folium.CircleMarker(
                    location=(lat, lon),
                    radius=4,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=imu_alpha,
                    opacity=imu_alpha,
                    weight=1,
                    popup=folium.Popup(
                        f"<b>{label_prefix} - device {d+1} - pos {i}</b><br>Lat: {lat:.6f}<br>Lon: {lon:.6f}<br>Depth: {_depth_str(depth)}",
                        max_width=200
                    ),
                    tooltip=f"{label_prefix} - device {d+1} - pos {i}"
                ).add_to(m)

    # --- Floaters: traiettoria tratteggiata per ciascun floater + etichetta sulla prima posizione ---
    if floaters_arr is not None and floaters_arr.size > 0:
        n_positions, n_floaters = floaters_arr.shape[0], floaters_arr.shape[1]

        for mi in range(n_floaters):
            floater_lats = floaters_arr[:, mi, 0]
            floater_lons = floaters_arr[:, mi, 1]
            if floaters_arr.shape[-1] >= 3:
                floater_depths = np.where(floaters_arr[:, mi, 2] == -999, np.nan, floaters_arr[:, mi, 2])
            else:
                floater_depths = np.full(n_positions, np.nan)

            traj_valid = [(lat, lon, depth)
                          for lat, lon, depth in zip(floater_lats, floater_lons, floater_depths)
                          if _is_valid(lat, lon)]

            if len(traj_valid) == 0:
                continue

            # --- Traiettoria tratteggiata (posizioni consecutive) ---
            if track_floaters and len(traj_valid) > 1:
                trajectory_coords = [(lat, lon) for lat, lon, _ in traj_valid]
                folium.PolyLine(
                    locations=trajectory_coords,
                    color="#FF0000",
                    weight=3,
                    opacity=0.6,
                    dash_array="5, 10",
                    tooltip=f"Traiettoria F{mi+1}"
                ).add_to(m)

            # --- Marker su tutte le posizioni tranne la prima (che ha già l'etichetta) ---
            for i, (lat, lon, depth) in enumerate(traj_valid[1:], start=2):
                folium.CircleMarker(
                    location=(lat, lon),
                    radius=4,
                    color="#FF0000",
                    fill=True,
                    fill_color="#FF0000",
                    fill_opacity=0.6,
                    weight=1,
                    popup=folium.Popup(
                        f"<b>{mi+1} - pos {i}</b><br>Lat: {lat:.6f}<br>Lon: {lon:.6f}<br>Depth: {_depth_str(depth)}",
                        max_width=180
                    ),
                    tooltip=f"{mi+1} - pos {i}"
                ).add_to(m)

            # --- Etichetta circolare sulla prima posizione nota ---
            first_lat, first_lon, first_depth = traj_valid[0]
            label = f"{mi+1}"
            folium.Marker(
                location=(first_lat, first_lon),
                popup=folium.Popup(
                    f"<b>{label}</b><br>Lat: {first_lat:.6f}<br>Lon: {first_lon:.6f}<br>Depth: {_depth_str(first_depth)}",
                    max_width=200
                ),
                tooltip=label,
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        background:#FF0000;
                        color:white;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        font-weight:bold;
                        font-family:monospace;
                        font-size:11px;
                        width:26px;
                        height:26px;
                        border-radius:50%;
                        border: 2px solid white;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.5);
                        white-space:nowrap;
                    ">{label}</div>""",
                    icon_size=(26, 26),
                    icon_anchor=(13, 13)
                )
            ).add_to(m)

    m.save(output_file)
    print(f"Map saved in: {output_file}")

    return m

if __name__ == '__main__':
    TX_Coordinates = np.load("Synth/TX_Coordinates.npy")
    RX_Coordinates = np.load("Synth/RX_Coordinates.npy")

    build_map(RX_Coordinates[0,:],
              TX_Coordinates,
              np.zeros([10,3]),
              "map.html",
              False,
              False,
              # Esempio di utilizzo dei nuovi parametri (forma attesa: (timestamps, device, 3)):
              # RX_fw_IMU=RX_fw_IMU_array,
              # RX_fw_IMU_MDS=RX_fw_IMU_MDS_array,
              # RX_bw_IMU=RX_bw_IMU_array,
              # RX_bw_IMU_MDS=RX_bw_IMU_MDS_array,
              )