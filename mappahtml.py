import folium
import numpy as np

# =============================================================================
# CONFIGURAZIONE - Modifica questi valori
# =============================================================================

# Punti principali H1 e H2 (lat, lon)
H1 = (45.4642, 9.1900)   # Milano
H2 = (44.4056, 8.9463)   # Genova

# Angoli misurati in senso antiorario partendo dall'Est (convenzione matematica)
# 0° = Est, 90° = Nord, 180° = Ovest, 270° = Sud
ANGLE_H1 = 120   # gradi
ANGLE_H2 = 60    # gradi

# Punti TX da visualizzare in giallo
LAT_TX  = [45.0703, 45.8400, 44.8015]
LONG_TX = [7.6869,  8.9300,  8.6150]

# Punti stimati da visualizzare in verde
LAT_ESTIMATED  = [45.2000, 44.6500]
LON_ESTIMATED  = [8.5000,  9.3000]

# =============================================================================
# FUNZIONI
# =============================================================================

def _is_valid(*values):
    """Restituisce True solo se nessuno dei valori è NaN o None."""
    return all(v is not None and not np.isnan(float(v)) for v in values)


def math_angle_to_bearing(angle_deg):
    """
    Converte un angolo matematico (antiorario da Est)
    nel bearing geografico (orario da Nord) usato nelle formule geodetiche.
    bearing = 90° - angle_math
    """
    bearing = 90.0 - angle_deg
    return bearing % 360


def destination_point(lat, lon, bearing_deg, distance_km):
    """
    Calcola il punto di destinazione dato un punto di partenza,
    un bearing (gradi, orario da Nord) e una distanza in km.
    Usa la formula di Haversine inversa.
    """
    R = 6371.0
    d = distance_km / R
    bearing = np.radians(bearing_deg)
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(d) +
        np.cos(lat1) * np.sin(d) * np.cos(bearing)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * np.sin(d) * np.cos(lat1),
        np.cos(d) - np.sin(lat1) * np.sin(lat2)
    )
    return (np.degrees(lat2), np.degrees(lon2))


def ray_intersection(p1, angle1_math, p2, angle2_math):
    """
    Calcola l'intersezione di due SEMIRETTE definite da:
      - p1, p2: punti di partenza (lat, lon)
      - angle1, angle2: angoli matematici (antiorario da Est, in gradi)

    Usa approssimazione piana (valida per distanze < ~500 km).

    Restituisce (lat, lon) dell'intersezione se esiste ed è nel verso
    positivo di entrambe le semirette, altrimenti None.
    """
    lat1, lon1 = p1
    lat2, lon2 = p2

    b1 = np.radians(angle1_math)
    b2 = np.radians(angle2_math)

    d1 = np.array([np.cos(b1), np.sin(b1)])
    d2 = np.array([np.cos(b2), np.sin(b2)])

    dp = np.array([lon2 - lon1, lat2 - lat1])

    denom = d1[0] * d2[1] - d1[1] * d2[0]

    if abs(denom) < 1e-10:
        print("Le semirette sono parallele: nessuna intersezione.")
        return None

    t = (dp[0] * d2[1] - dp[1] * d2[0]) / denom
    s = (dp[0] * d1[1] - dp[1] * d1[0]) / denom

    if t < 0 or s < 0:
        print(f"Intersezione calcolata (t={t:.3f}, s={s:.3f}) ma cade nel verso opposto delle semirette.")
        return None

    ix_lon = lon1 + t * d1[0]
    ix_lat = lat1 + t * d1[1]
    return (ix_lat, ix_lon)


def build_map(h1, h2, angle_h1, angle_h2, lat_tx, long_tx,
              lat_estimated, lon_estimated,
              output_file="mappa_intersezione.html"):
    """
    Costruisce, salva e restituisce la mappa Folium con:
      - H1, H2 come marker con etichetta colorata
      - Punti TX in giallo  (NaN ignorati)
      - Punti stimati in verde (NaN ignorati)
      - Punto di intersezione delle semirette in verde brillante (se esiste)
    """

    # --- Filtra i NaN dagli array prima di qualsiasi elaborazione ---
    tx_valid  = [(lat, lon) for lat, lon in zip(lat_tx, long_tx)
                 if _is_valid(lat, lon)]
    est_valid = [(lat, lon) for lat, lon in zip(lat_estimated, lon_estimated)
                 if _is_valid(lat, lon)]

    skipped_tx  = len(list(zip(lat_tx, long_tx))) - len(tx_valid)
    skipped_est = len(list(zip(lat_estimated, lon_estimated))) - len(est_valid)
    if skipped_tx:
        print(f"⚠️  Ignorati {skipped_tx} punti TX con coordinate NaN.")
    if skipped_est:
        print(f"⚠️  Ignorati {skipped_est} punti stimati con coordinate NaN.")

    # Centro mappa: media di tutte le coordinate valide
    all_lats = [h1[0], h2[0]] + [p[0] for p in tx_valid] + [p[0] for p in est_valid]
    all_lons = [h1[1], h2[1]] + [p[1] for p in tx_valid] + [p[1] for p in est_valid]
    center = [np.mean(all_lats), np.mean(all_lons)]

    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB dark_matter")

    # --- Punti TX (gialli) ---
    for i, (lat, lon) in enumerate(tx_valid):
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

    # --- Marker H1 e H2 ---
    stations = [
        {"point": h1, "angle": angle_h1, "label": "H1", "color": "#FF4444"},
        {"point": h2, "angle": angle_h2, "label": "H2", "color": "#4488FF"},
    ]

    for s in stations:
        pt    = s["point"]
        angle = s["angle"]
        color = s["color"]
        label = s["label"]

        folium.Marker(
            location=pt,
            popup=folium.Popup(
                f"<b>{label}</b><br>Lat: {pt[0]}<br>Lon: {pt[1]}<br>Angolo: {angle}°",
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

    # --- Intersezione delle semirette ---
    intersection = ray_intersection(h1, angle_h1, h2, angle_h2)

    if intersection:
        ilat, ilon = intersection
        print(f"✅ Intersezione trovata: lat={ilat:.5f}, lon={ilon:.5f}")

        folium.CircleMarker(
            location=intersection,
            radius=14,
            color="#00FF88",
            fill=False,
            weight=2,
            opacity=0.5,
        ).add_to(m)

        folium.CircleMarker(
            location=intersection,
            radius=8,
            color="#00FF88",
            fill=True,
            fill_color="#00FF88",
            fill_opacity=1.0,
            weight=2,
            popup=folium.Popup(
                f"<b>Intersezione</b><br>Lat: {ilat:.5f}<br>Lon: {ilon:.5f}",
                max_width=200
            ),
            tooltip="Intersezione H1 ∩ H2"
        ).add_to(m)

    else:
        print("⚠️  Nessuna intersezione valida tra le semirette.")

    # --- Salvataggio ---
    m.save(output_file)
    print(f"🗺️  Mappa salvata in: {output_file}")

    return m


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    build_map(
        h1=H1,
        h2=H2,
        angle_h1=ANGLE_H1,
        angle_h2=ANGLE_H2,
        lat_tx=LAT_TX,
        long_tx=LONG_TX,
        lat_estimated=LAT_ESTIMATED,
        lon_estimated=LON_ESTIMATED,
        output_file="mappa_intersezione.html"
    )