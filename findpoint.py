import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from collections import Counter
import math


def math_to_bearing(math_angle_deg: float) -> float:
    """
    Converte un angolo matematico (antiorario da EST, in gradi)
    nel bearing geografico (orario da NORD, in gradi) usato internamente
    per i calcoli geodetici.

    Relazione:  bearing = 90 - math_angle   (mod 360)
    """
    return (90.0 - math_angle_deg) % 360.0

def _flat_earth_intersection(
    lat1: float, lon1: float, brg1: float,
    lat2: float, lon2: float, brg2: float,
) -> tuple[float, float]:
    """
    Punto di intersezione tra due semirette, approssimazione piana.

    Valida per distanze < ~10 km (errore < qualche metro).
    Per distanze maggiori usa _great_circle_intersection.

    Parametri
    ---------
    lat1, lon1 : coordinate del primo punto (gradi decimali)
    brg1       : bearing geografico (°N, orario)
    lat2, lon2 : coordinate del secondo punto (gradi decimali)
    brg2       : bearing geografico (°N, orario)

    Ritorna
    -------
    (lat, lon) del punto di intersezione, oppure (nan, nan) se non esiste.
    """
    # Fattore di scala per la longitudine nel piano locale
    cos_lat = math.cos(math.radians((lat1 + lat2) / 2))

    # Bearing → vettore direzione nel piano (x=Est, y=Nord)
    def brg_to_vec(brg_deg):
        r = math.radians(brg_deg)
        return math.sin(r), math.cos(r)   # (dx, dy)

    dx1, dy1 = brg_to_vec(brg1)
    dx2, dy2 = brg_to_vec(brg2)

    # Coordinate nel piano locale (gradi, con longitudine scalata)
    x1, y1 = lon1 * cos_lat, lat1
    x2, y2 = lon2 * cos_lat, lat2

    # Intersezione di due rette parametriche:
    #   P1 + t * d1 = P2 + s * d2
    # Risolto per t con la regola di Cramer
    denom = dx1 * dy2 - dy1 * dx2 # Determinante

    if abs(denom) < 1e-12:      # rette parallele o coincidenti
        return math.nan, math.nan

    t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denom

    if t < 0:                   # intersezione dietro la semiretta 1
        return math.nan, math.nan

    s = ((x2 - x1) * dy1 - (y2 - y1) * dx1) / denom
    if s < 0:                   # intersezione dietro la semiretta 2
        return math.nan, math.nan

    # Coordinate geografiche del punto di intersezione
    lon_out = (x1 + t * dx1) / cos_lat
    lat_out =  y1 + t * dy1

    return lat_out, lon_out

def _least_squares_point_n(
    lats: np.ndarray,
    lons: np.ndarray,
    brgs: np.ndarray,
) -> tuple[float, float] | None:
    """
    Trova il punto che minimizza la somma delle distanze al quadrato
    dalle N semirette geodetiche (approssimazione flat-earth).

    Parametri
    ---------
    lats, lons : array (N,) di coordinate degli idrofoni in gradi.
    brgs       : array (N,) di bearing geografici in gradi (CW da Nord).

    Ritorna
    -------
    (lat, lon) del punto ottimale, oppure None se il sistema è singolare.
    """
    # Direzioni unitarie delle rette in coordinate (dx=Est, dy=Nord)
    brgs_rad = np.deg2rad(brgs)
    dx = np.sin(brgs_rad)  # componente Est
    dy = np.cos(brgs_rad)  # componente Nord

    # Sistema ai minimi quadrati: minimizza sum_i dist(P, retta_i)^2
    # La distanza dal punto P=(x,y) alla retta passante per (x0,y0)
    # con direzione (dx,dy) è: |(P - H) x d| = (dy*(x-x0) - dx*(y-y0))
    # Matrice A e vettore b del sistema normale A^T A p = A^T b
    # con proiezione ortogonale: (I - d d^T) P = (I - d d^T) H

    # Usa coordinate metriche approssimate centrate sulla media degli idrofoni
    lat0 = np.mean(lats)
    lon0 = np.mean(lons)
    R = 6371000.0  # raggio terrestre in metri
    lat0_rad = np.deg2rad(lat0)

    # Converte lat/lon -> metri relativi al centro
    x0 = np.deg2rad(lons - lon0) * R * np.cos(lat0_rad)
    y0 = np.deg2rad(lats - lat0) * R

    # Proiettori ortogonali: per ogni retta i, (I - d_i d_i^T)
    # A x = b  =>  sum_i (I - d_i d_i^T) @ p = sum_i (I - d_i d_i^T) @ h_i
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for i in range(len(lats)):
        d = np.array([dx[i], dy[i]])
        P_orth = np.eye(2) - np.outer(d, d)
        h = np.array([x0[i], y0[i]])
        A += P_orth
        b += P_orth @ h

    try:
        p = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    if not np.all(np.isfinite(p)):
        return None

    # Riconverte metri -> gradi
    lon_opt = lon0 + np.rad2deg(p[0] / (R * np.cos(lat0_rad)))
    lat_opt = lat0 + np.rad2deg(p[1] / R)

    return float(lat_opt), float(lon_opt)

def _flat_dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distanza in metri tra due punti vicini, approssimazione flat-earth."""
    meters_per_deg_lat = 111_319.9
    meters_per_deg_lon = 111_319.9 * np.cos(np.deg2rad((lat1 + lat2) / 2))
    dy = (lat2 - lat1) * meters_per_deg_lat
    dx = (lon2 - lon1) * meters_per_deg_lon
    return np.sqrt(dx**2 + dy**2)


def find_points(
    hydrophones: np.ndarray,
    bearings: np.ndarray,
    elevation_array: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per ogni colonna di bearing, calcola il punto che minimizza la distanza
    dalle N semirette geodetiche emesse dagli N idrofoni.
    Se hydrophones ha una terza colonna (profondità) ed elevation_array non è None,
    stima anche la profondità del target.

    Parametri
    ---------
    hydrophones : np.ndarray di forma (N, 2) o (N, 3)
        Coordinate [lat, lon] o [lat, lon, depth_m] in gradi decimali (depth in metri,
        positivo = sotto la superficie).
    bearings : np.ndarray di forma (N, M)
        M set di N angoli (gradi), convenzione CCW da EST.
        Ogni colonna m contiene i bearing dei N idrofoni per l'evento m.
    elevation_array : np.ndarray di forma (N, M), opzionale
        Angoli di elevazione in gradi per ogni idrofono e ogni evento.
        Positivo = sopra l'orizzonte, negativo = sotto.
        Ignorato se hydrophones non contiene la profondità.

    Ritorna
    -------
    positions : np.ndarray di forma (M, 2)
        Colonne [latitudine, longitudine]. Righe senza soluzione → [NaN, NaN].
    depths : np.ndarray di forma (M,)
        Profondità stimata in metri (positivo = sotto superficie) se
        hydrophones ha 3 colonne ed elevation_array non è None,
        altrimenti array di -999.
    """
    hydrophones = np.asarray(hydrophones, dtype=float)
    bearings    = np.asarray(bearings,    dtype=float)

    if hydrophones.ndim != 2 or hydrophones.shape[1] not in (2, 3):
        raise ValueError(
            f"hydrophones deve avere forma (N, 2) o (N, 3), ma ha forma {hydrophones.shape}."
        )
    if bearings.ndim != 2:
        raise ValueError(
            f"bearings deve avere forma (N, M), ma ha forma {bearings.shape}."
        )

    n_hydro  = hydrophones.shape[0]
    n_events = bearings.shape[1]

    if bearings.shape[0] != n_hydro:
        raise ValueError(
            f"bearings deve avere {n_hydro} righe (una per idrofono), "
            f"ma ne ha {bearings.shape[0]}."
        )
    if n_hydro < 2:
        raise ValueError("Servono almeno 2 idrofoni.")

    # ── Controlla se è disponibile la profondità degli idrofoni ──────────
    has_depth = hydrophones.shape[1] == 3
    use_elevation = has_depth and elevation_array is not None

    lats = hydrophones[:, 0]
    lons = hydrophones[:, 1]
    hydrophone_depths = hydrophones[:, 2] if has_depth else None

    if use_elevation:
        elevation_array = np.asarray(elevation_array, dtype=float)
        if elevation_array.shape != bearings.shape:
            raise ValueError(
                f"elevation_array deve avere la stessa forma di bearings {bearings.shape}, "
                f"ma ha forma {elevation_array.shape}."
            )

    positions = np.full((n_events, 3), np.nan)


    brgs = np.vectorize(math_to_bearing)(bearings)  # (N, M)

    for m in range(n_events):
        brg_m = brgs[:, m]  # (N,) bearing per l'evento m

        # ── Stima lat/lon ─────────────────────────────────────────────────
        if n_hydro == 2:
            lat_i, lon_i = _flat_earth_intersection(
                lats[0], lons[0], brg_m[0],
                lats[1], lons[1], brg_m[1],
            )
            if not np.isnan(lat_i):
                positions[m, 0] = lat_i
                positions[m, 1] = lon_i
        else:
            candidates = []
            for i in range(n_hydro):
                for j in range(i + 1, n_hydro):
                    lat_i, lon_i = _flat_earth_intersection(
                        lats[i], lons[i], brg_m[i],
                        lats[j], lons[j], brg_m[j],
                    )
                    if not np.isnan(lat_i):
                        candidates.append((lat_i, lon_i))

            opt = _least_squares_point_n(lats, lons, brg_m)

            if opt is None:
                if len(candidates) >= 2:
                    opt = (
                        float(np.mean([p[0] for p in candidates])),
                        float(np.mean([p[1] for p in candidates])),
                    )
                elif len(candidates) == 1:
                    opt = candidates[0]
                else:
                    continue

            positions[m, 0] = opt[0]
            positions[m, 1] = opt[1]

        # ── Stima profondità (solo se use_elevation e posizione valida) ───
        if use_elevation and not np.isnan(positions[m, 0]):
            depth_estimates = []
            for i in range(n_hydro):
                el_rad  = np.deg2rad(elevation_array[i, m])
                dist_h  = _flat_dist_m(lats[i], lons[i], positions[m, 0], positions[m, 1])
                delta_z = dist_h * np.tan(el_rad)
                # el < 0 → target più in basso dell'idrofono → profondità maggiore
                depth_estimates.append(hydrophone_depths[i] - delta_z)
            positions[m,2] = float(np.mean(depth_estimates))

    return positions