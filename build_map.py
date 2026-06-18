import datetime as DT
import folium
import folium.plugins
import numpy as np
from folium import MacroElement
from jinja2 import Template
import csv

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

def build_map(floaters_coordinates, TX_positions_coordinates, estimated_vessel_coordinates, output_file, track_trajectory=False):
    """
    Costruisce, salva e restituisce la mappa Folium con:
      - Floaters (H1, H2, ...) come marker con etichetta rossa
      - Punti TX in giallo  (NaN ignorati)
      - Punti stimati in verde (NaN ignorati)
      - Traiettoria opzionale che collega i punti stimati sequenzialmente

    Per tutti e tre gli array di coordinate, la profondità è opzionale:
    se presente come terza colonna viene mostrata nel popup, altrimenti "N/A".
    I valori -999 sono trattati come profondità non disponibile.

    Parametri
    ---------
    floaters_coordinates : array-like di forma (N, 2) o (N, 3)
    TX_positions_coordinates : array-like di forma (M, 2) o (M, 3)
    estimated_vessel_coordinates : array-like di forma (K, 2) o (K, 3)
    output_file : str
    track_trajectory : bool, opzionale (default=False)
        Se True, mostra una linea che collega i punti stimati in ordine cronologico/sequenziale.
    """

    def _extract_coords(arr):
        """Restituisce (lats, lons, depths) dove depths è NaN se non disponibile."""
        arr = np.asarray(arr, dtype=float)
        lats = arr[:, 0]
        lons = arr[:, 1]
        if arr.shape[1] >= 3:
            depths = arr[:, 2]
            depths = np.where(depths == -999, np.nan, depths)
        else:
            depths = np.full(len(lats), np.nan)
        return lats, lons, depths

    def _depth_str(depth):
        """Formatta la profondità per il popup."""
        return f"{depth:.1f} m" if not np.isnan(depth) else "N/A"

    lats_fl,  lons_fl,  depths_fl  = _extract_coords(floaters_coordinates)
    lats_tx,  lons_tx,  depths_tx  = _extract_coords(TX_positions_coordinates)
    lats_est, lons_est, depths_est = _extract_coords(estimated_vessel_coordinates)

    tx_valid  = [(lat, lon, depth)
                 for lat, lon, depth in zip(lats_tx, lons_tx, depths_tx)
                 if _is_valid(lat, lon)]

    est_valid = [(lat, lon, depth)
                 for lat, lon, depth in zip(lats_est, lons_est, depths_est)
                 if _is_valid(lat, lon)]

    # Creazione mappa (centrata su H1)
    m = folium.Map(location=(lats_fl[0], lons_fl[0]), zoom_start=15, tiles="OpenStreetMap")

    ScaleBar().add_to(m)

    folium.plugins.MeasureControl(
        position="bottomleft",
        primary_length_unit="meters",
        secondary_length_unit="kilometers",
        primary_area_unit="sqmeters",
        secondary_area_unit="sqkilometers"
    ).add_to(m)

    # --- Punti TX (gialli) ---
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

    # --- Traiettoria (opzionale) ---
    # Viene disegnata prima dei punti stimati in modo che i marker rimangano visivamente "sopra" la linea
    if track_trajectory and len(est_valid) > 1:
        trajectory_coords = [(lat, lon) for lat, lon, _ in est_valid]
        folium.PolyLine(
            locations=trajectory_coords,
            color="#00CC66",
            weight=5,
            opacity=0.7,
            dash_array="5, 10", # Stile tratteggiato elegante, rimuovilo se preferisci la linea continua
            tooltip="Traiettoria Stimata Vessel"
        ).add_to(m)

    # --- Punti stimati (verdi) ---
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

    # --- Floaters ---
    for i, (lat, lon, depth) in enumerate(zip(lats_fl, lons_fl, depths_fl)):
        label = f"H{i+1}"
        folium.Marker(
            location=(lat, lon),
            popup=folium.Popup(
                f"<b>{label}</b><br>Lat: {lat:.6f}<br>Lon: {lon:.6f}<br>Depth: {_depth_str(depth)}",
                max_width=200
            ),
            tooltip=label,
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    background:#FF0000;
                    color:white;
                    font-weight:bold;
                    font-family:monospace;
                    font-size:13px;
                    padding:4px 8px;
                    border-radius:4px;
                    border: 2px solid white;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.5);
                    white-space:nowrap;
                ">{label}</div>""",
                icon_size=(40, 28),
                icon_anchor=(20, 14)
            )
        ).add_to(m)

    m.save(output_file)
    print(f"Map saved in: {output_file}")

    return m