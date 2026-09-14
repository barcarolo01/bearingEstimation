import folium
import folium.plugins
import numpy as np
from folium import MacroElement
from jinja2 import Template

from maps.map_common import (
    Track,
    as_multi,
    as_points,
    depth_str,
    first_valid_location,
    normalize_tracks,
    valid_points,
)

FLOATER_COLOR = "#FF0000"
TX_COLOR = "#FFD700"
EST_COLOR = "#00CC66"

# OpenStreetMap basemap.
#
# IMPORTANT:
# The standard OSM tile service should be used with a normal HTTP(S)
# application context. If map.html is opened directly with file://, some
# browsers/proxies may omit the Referer and the OSM tile server can respond
# with HTTP 403. The Python code below uses the official OSM tile URL; if your
# browser still shows 403, serve map.html through a small local HTTP server
# (see the note below the file).
BASEMAP_TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
BASEMAP_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">'
    'OpenStreetMap</a> contributors'
)

class ScaleBar(MacroElement):
    """Fixed scale bar, automatically updated on zoom."""

    _template = Template("""
        {% macro script(this, kwargs) %}

        // Create the scale bar container
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

            // Width of the map in pixels
            var leftPoint = map.latLngToContainerPoint(
                L.latLng(center.lat, bounds.getWest())
            );
            var rightPoint = map.latLngToContainerPoint(
                L.latLng(center.lat, bounds.getEast())
            );
            var pixelWidth = rightPoint.x - leftPoint.x;

            // Real width of the map in meters
            var realWidth = center.distanceTo(
                L.latLng(center.lat, bounds.getEast())
            ) * 2;

            // Scale: meters per pixel
            var metersPerPixel = realWidth / pixelWidth;

            // Target bar width (100px) -> real distance
            var targetPixels = 100;
            var targetMeters = metersPerPixel * targetPixels;

            // Round to a "nice" number
            var magnitude = Math.pow(10, Math.floor(Math.log10(targetMeters)));
            var nice = [1, 2, 5, 10];
            var niceMeters = magnitude;
            for (var i = 0; i < nice.length; i++) {
                if (nice[i] * magnitude >= targetMeters * 0.5) {
                    niceMeters = nice[i] * magnitude;
                    break;
                }
            }

            // Actual pixels for the rounded distance
            var barPixels = niceMeters / metersPerPixel;

            // Label
            var label;
            if (niceMeters >= 1000) {
                label = (niceMeters / 1000) + ' km';
            } else {
                label = niceMeters + ' m';
            }

            // Update the DOM
            var div = document.querySelector('.scale-bar');
            if (div) {
                div.style.width = barPixels + 'px';
                div.style.minWidth = 'unset';
                div.innerHTML = label;
            }
        }

        // Update on creation and on every zoom/pan
        {{ this._parent.get_name() }}.on('zoomend moveend load', updateScale);
        setTimeout(updateScale, 300);

        {% endmacro %}
    """)

    def __init__(self):
        super().__init__()


def build_folium_map(
    floater_coordinates=None,
    TX_coordinates=None,
    estimated_vessel_coordinates=None,
    tracks=None,
    output_file="map.html",
    track_alpha=0.7,
    zoom_start=15,
):
    """
    Build, save and return the folium map.

    Takes the same parameters as `build_local_cartesian_map`, except
    `center_coordinates`, `window_width_m` and `window_height_m` (the folium map
    centers itself on the first available valid point).

    floater_coordinates : array-like (N, M, 2|3) or (M, 2|3), or None
        N = time steps, M = number of floaters, columns [lat, lon, (depth)].
        Each floater has its own trajectory: different floaters are never connected.
    TX_coordinates : array-like (K, 2|3), or None
    estimated_vessel_coordinates : array-like (K, 2|3), or None
    tracks : Track | dict | tuple | sequence of those, or None
        Generic series of points: name (str), color (str) and points of shape
        (NUMBER_STEPS, 2|3) or (NUMBER_STEPS, M, 2|3). With the 3-dimensional
        shape, M independent series sharing name and color are drawn: points are
        connected along the step axis only, never across different indices.
    output_file : str
    track_alpha : float
        Transparency of the series in `tracks`.
    zoom_start : int

    Depth: the depth column is optional; -999 values are treated as missing.
    """

    # --- Normalize every input to a single [lat, lon, depth] layout ---
    floaters = as_multi(floater_coordinates)                            # (N, M, 3)
    tx = valid_points(as_points(TX_coordinates))                        # (K, 3)
    estimated = valid_points(as_points(estimated_vessel_coordinates))   # (K, 3)
    track_list = normalize_tracks(tracks)                               # each .xyz is (N, M, 3)

    center = first_valid_location(floaters, tx, estimated, *[t.xyz for t in track_list])
    if center is None:
        raise ValueError("No valid coordinate provided to center the map.")

    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles=BASEMAP_TILES,
        attr=BASEMAP_ATTRIBUTION,
        max_zoom=19,
    )

    ScaleBar().add_to(m)
    folium.plugins.MeasureControl(
        position="bottomleft",
        primary_length_unit="meters",
        secondary_length_unit="kilometers",
        primary_area_unit="sqmeters",
        secondary_area_unit="sqkilometers",
    ).add_to(m)

    # --- TX points (yellow) ---
    if len(tx) > 1:
        folium.PolyLine(
            locations=[(lat, lon) for lat, lon, _ in tx],
            color=TX_COLOR, weight=5, opacity=0.7,
            tooltip="TX trajectory",
        ).add_to(m)

    for i, (lat, lon, depth) in enumerate(tx, start=1):
        folium.CircleMarker(
            location=(lat, lon), radius=10,
            color=TX_COLOR, fill=True, fill_color=TX_COLOR, fill_opacity=0.9, weight=2,
            popup=folium.Popup(
                f"<b>TX {i}</b><br>Lat: {lat:.6f}<br>Lon: {lon:.6f}<br>Depth: {depth_str(depth)}",
                max_width=180),
            tooltip=f"TX {i}",
        ).add_to(m)

    # --- Estimated positions (green) ---
    if len(estimated) > 1:
        folium.PolyLine(
            locations=[(lat, lon) for lat, lon, _ in estimated],
            color=EST_COLOR, weight=5, opacity=0.7, dash_array="5, 10",
            tooltip="Estimated vessel trajectory",
        ).add_to(m)

    for i, (lat, lon, depth) in enumerate(estimated, start=1):
        folium.CircleMarker(
            location=(lat, lon), radius=7,
            color=EST_COLOR, fill=True, fill_color=EST_COLOR, fill_opacity=0.9, weight=2,
            popup=folium.Popup(
                f"<b>Estimated {i}</b><br>Lat: {lat:.6f}<br>Lon: {lon:.6f}<br>"
                f"Depth: {depth_str(depth)}",
                max_width=180),
            tooltip=f"Estimated {i}",
        ).add_to(m)

    # --- Generic tracks: one polyline per index, never connected to each other ---
    for track in track_list:
        if track.xyz.size == 0:
            continue

        n_series = track.xyz.shape[1]
        for series_index in range(n_series):
            # series `series_index`, in step order
            points = valid_points(track.xyz[:, series_index, :])
            if len(points) == 0:
                continue

            label = track.name if n_series == 1 else f"{track.name} [{series_index + 1}]"

            if len(points) > 1:
                folium.PolyLine(
                    locations=[(lat, lon) for lat, lon, _ in points],
                    color=track.color, weight=3, opacity=track_alpha,
                    tooltip=label,
                ).add_to(m)

            for i, (lat, lon, depth) in enumerate(points, start=1):
                folium.CircleMarker(
                    location=(lat, lon), radius=4,
                    color=track.color, fill=True, fill_color=track.color,
                    fill_opacity=track_alpha, opacity=track_alpha, weight=1,
                    popup=folium.Popup(
                        f"<b>{label} - pos {i}</b><br>Lat: {lat:.6f}<br>"
                        f"Lon: {lon:.6f}<br>Depth: {depth_str(depth)}",
                        max_width=200),
                    tooltip=f"{label} - pos {i}",
                ).add_to(m)

    # --- Floaters: dashed trajectory + label on the first known position ---
    if floaters.size > 0:
        for floater_index in range(floaters.shape[1]):
            trajectory = valid_points(floaters[:, floater_index, :])
            if len(trajectory) == 0:
                continue

            if len(trajectory) > 1:
                folium.PolyLine(
                    locations=[(lat, lon) for lat, lon, _ in trajectory],
                    color=FLOATER_COLOR, weight=3, opacity=0.6, dash_array="5, 10",
                    tooltip=f"F{floater_index + 1} trajectory",
                ).add_to(m)

            # Markers on every position except the first one, which carries the label
            for i, (lat, lon, depth) in enumerate(trajectory[1:], start=2):
                folium.CircleMarker(
                    location=(lat, lon), radius=4,
                    color=FLOATER_COLOR, fill=True, fill_color=FLOATER_COLOR,
                    fill_opacity=0.6, weight=1,
                    popup=folium.Popup(
                        f"<b>{floater_index + 1} - pos {i}</b><br>Lat: {lat:.6f}<br>"
                        f"Lon: {lon:.6f}<br>Depth: {depth_str(depth)}",
                        max_width=180),
                    tooltip=f"{floater_index + 1} - pos {i}",
                ).add_to(m)

            first_lat, first_lon, first_depth = trajectory[0]
            label = f"{floater_index + 1}"
            folium.Marker(
                location=(first_lat, first_lon),
                popup=folium.Popup(
                    f"<b>{label}</b><br>Lat: {first_lat:.6f}<br>"
                    f"Lon: {first_lon:.6f}<br>Depth: {depth_str(first_depth)}",
                    max_width=200),
                tooltip=label,
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        background:{FLOATER_COLOR};
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
                    icon_anchor=(13, 13),
                ),
            ).add_to(m)

    m.save(output_file)
    print(f"Map saved in: {output_file}")
    print("If OSM tiles return HTTP 403 when opening the HTML directly, serve "
          "the file over HTTP, e.g.: python -m http.server 8000")

    return m


# Alias kept for backwards compatibility with the old name
build_map = build_folium_map


# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    RX_Coordinates = np.load("Synth/RX_Coordinates.npy")
    TX_Coordinates = np.load("Synth/TX_Coordinates.npy")
    Est_Coordinates = np.load("Synth/Estimated_Coordinates.npy")

    build_folium_map(
        floater_coordinates=RX_Coordinates,
        TX_coordinates=TX_Coordinates,
        estimated_vessel_coordinates=Est_Coordinates,
        tracks=[
            Track("RX IMU", np.load("Synth/RX_fw_IMU.npy"), "#0000FF"),
            Track("RX IMU+MDS", np.load("Synth/RX_fw_IMU_MDS.npy"), "#2AB040"),
            Track("Compensated", np.load("Synth/RX_bw_IMU.npy"), "#FF8822"),
        ],
        output_file="map.html",
    )
