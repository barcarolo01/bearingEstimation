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

def build_map(floaters_coordinates, TX_positions_coordinates, estimated_vessel_coordinates, output_file):
    """
    Costruisce, salva e restituisce la mappa Folium con:
      - H1, H2 come marker con etichetta colorata
      - Punti TX in giallo  (NaN ignorati)
      - Punti stimati in verde (NaN ignorati)
    """

    lat_tx  = TX_positions_coordinates[:,0]
    lon_tx = TX_positions_coordinates[:,1]
    lat_estimated = estimated_vessel_coordinates[:,0]
    lon_estimated = estimated_vessel_coordinates[:,1]

    # Elimina i NaN dagli array
    tx_valid  = [(lat, lon) for lat, lon in zip(lat_tx, lon_tx)
                 if _is_valid(lat, lon)]
    est_valid = [(lat, lon) for lat, lon in zip(lat_estimated, lon_estimated)
                 if _is_valid(lat, lon)]

    
    # Creazione mapap (centrata su H1)
    m = folium.Map(location=floaters_coordinates[0], zoom_start=15, tiles="OpenStreetMap")

    # Aggiungi scala
    ScaleBar().add_to(m)

    # Aggiungi strumento di misura
    folium.plugins.MeasureControl(
        position="bottomleft",
        primary_length_unit="meters",
        secondary_length_unit="kilometers",
        primary_area_unit="sqmeters",
        secondary_area_unit="sqkilometers"
    ).add_to(m)

    # --- Punti TX (gialli) ---
    for i, (lat, lon) in enumerate(tx_valid):
        print(i)
        folium.CircleMarker(
            location=(lat, lon),
            radius=7,
            color="#FFD700",
            fill=True,
            fill_color="#FFD700",
            fill_opacity=0.9,
            weight=2,
            popup=folium.Popup(f"<b>TX {i+1}</b><br>Lat: {lat}<br>Lon: {lon}", max_width=180),
            tooltip=f"TX {i+1}"
        ).add_to(m)

    # --- Punti stimati (verdi) ---
    for i, (lat, lon) in enumerate(est_valid):
        folium.CircleMarker(
            location=(lat, lon),
            radius=7,
            color="#00CC66",
            fill=True,
            fill_color="#00CC66",
            fill_opacity=0.9,
            weight=2,
            popup=folium.Popup(f"<b>Stimato {i+1}</b><br>Lat: {lat}<br>Lon: {lon}", max_width=180),
            tooltip=f"Stimato {i+1}"
        ).add_to(m)

    # Floaters
    floaters = []
    for i in range(len(floaters_coordinates)):
        floaters.append({"point": floaters_coordinates[i], "label": f"H{i+1}", "color": "#FF0000"})

    for s in floaters:
        pt    = s["point"]
        color = s["color"]
        label = s["label"]

        folium.Marker(
            location=pt,
            popup=folium.Popup(
                f"<b>{label}</b><br>Lat: {pt[0]}<br>Lon: {pt[1]}",
                max_width=200
            ),
            tooltip=label,
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    background:{color};
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

    # --- Salvataggio ---
    m.save(output_file)
    print(f"🗺️  Mappa salvata in: {output_file}")

    return m


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    TX_positions_coordinates = np.load("Synth/TX_Coordinates.npy")
    floaters_coordinates = np.load("Synth/RX_Coordinates.npy")
    Estimated_positions = np.load("Synth/Estimated_positions.npy")
    build_map(
        floaters_coordinates = floaters_coordinates,
        TX_positions_coordinates = TX_positions_coordinates,
        estimated_vessel_coordinates = Estimated_positions,
        output_file="map.html"
    )