import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
import math

"""
find_intersection_points.py

Calcola i punti di intersezione tra semirette geodetiche sulla superficie terrestre.
Gli angoli sono espressi in gradi, misurati in senso ANTIORARIO a partire dalla direzione EST
(convenzione matematica standard).

Uso:
    H1 = (-35, 153)
    H2 = (-33, 151)
    bearing_H1 = [45, 90, 135]
    bearing_H2 = [225, 270, 315]
    result = find_points(H1, H2, bearing_H1, bearing_H2)
    # result è un array Nx2 con colonne [lat, lon]
"""

# ---------------------------------------------------------------------------
# Utilità angolari
# ---------------------------------------------------------------------------

def math_to_bearing(math_angle_deg: float) -> float:
    """
    Converte un angolo matematico (antiorario da EST, in gradi)
    nel bearing geografico (orario da NORD, in gradi) usato internamente
    per i calcoli geodetici.

    Relazione:  bearing = 90 - math_angle   (mod 360)
    """
    return (90.0 - math_angle_deg) % 360.0


# ---------------------------------------------------------------------------
# Intersezione di due grandi cerchi  (coordinate sferiche)
# ---------------------------------------------------------------------------

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

def _great_circle_intersection(
    lat1: float, lon1: float, brg1: float,
    lat2: float, lon2: float, brg2: float,
) -> tuple[float, float] | tuple[float, float]:
    """
    Restituisce il punto di intersezione tra due semirette geodetiche.

    Parametri
    ---------
    lat1, lon1 : coordinate del primo punto (gradi decimali)
    brg1       : bearing geografico (°N, orario) della prima semiretta
    lat2, lon2 : coordinate del secondo punto (gradi decimali)
    brg2       : bearing geografico (°N, orario) della seconda semiretta

    Ritorna
    -------
    (lat, lon) del punto di intersezione, oppure (nan, nan) se non esiste.

    Algoritmo
    ---------
    Implementazione vettoriale basata sul metodo di Ed Williams
    (Aviation Formulary, intersection of two radials):
    https://edwilliams.org/avform147.htm#Intersection
    """
    # Converti tutto in radianti
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    brg1_rad = np.radians(brg1)
    brg2_rad = np.radians(brg2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Distanza angolare tra i due punti di origine (haversine)
    angular_dist_12 = 2.0 * np.arcsin(
        np.sqrt(
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        )
    )

    if np.abs(angular_dist_12) < 1e-12:    # punti coincidenti
        return np.nan, np.nan

    # Bearing iniziale e finale del segmento 1→2
    cos_brg_fwd = (
        (np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(angular_dist_12))
        / (np.sin(angular_dist_12) * np.cos(lat1_rad))
    )
    cos_brg_rev = (
        (np.sin(lat1_rad) - np.sin(lat2_rad) * np.cos(angular_dist_12))
        / (np.sin(angular_dist_12) * np.cos(lat2_rad))
    )

    cos_brg_fwd = np.clip(cos_brg_fwd, -1.0, 1.0)
    cos_brg_rev = np.clip(cos_brg_rev, -1.0, 1.0)

    brg_fwd = np.arccos(cos_brg_fwd)   # bearing P1→P2 visto da P1
    brg_rev = np.arccos(cos_brg_rev)   # bearing P2→P1 visto da P2

    if np.sin(dlon) > 0:
        brg_12, brg_21 = brg_fwd, 2.0 * np.pi - brg_rev
    else:
        brg_12, brg_21 = 2.0 * np.pi - brg_fwd, brg_rev

    angle_at_p1 = brg1_rad - brg_12    # angolo al vertice P1
    angle_at_p2 = brg_21   - brg2_rad  # angolo al vertice P2

    # Angoli quasi paralleli → nessuna intersezione utile
    if np.abs(np.sin(angle_at_p1)) < 1e-12 and np.abs(np.sin(angle_at_p2)) < 1e-12:
        return np.nan, np.nan

    angle_at_p3 = np.arccos(
        np.clip(
            -np.cos(angle_at_p1) * np.cos(angle_at_p2)
            + np.sin(angle_at_p1) * np.sin(angle_at_p2) * np.cos(angular_dist_12),
            -1.0, 1.0,
        )
    )

    angular_dist_13 = np.arctan2(
        np.sin(angular_dist_12) * np.sin(angle_at_p1) * np.sin(angle_at_p2),
        np.cos(angle_at_p2) + np.cos(angle_at_p1) * np.cos(angle_at_p3),
    )

    lat3_rad = np.arcsin(
        np.clip(
            np.sin(lat1_rad) * np.cos(angular_dist_13)
            + np.cos(lat1_rad) * np.sin(angular_dist_13) * np.cos(brg1_rad),
            -1.0, 1.0,
        )
    )

    dlon_13 = np.arctan2(
        np.sin(brg1_rad) * np.sin(angular_dist_13) * np.cos(lat1_rad),
        np.cos(angular_dist_13) - np.sin(lat1_rad) * np.sin(lat3_rad),
    )

    lon3_rad = lon1_rad + dlon_13

    lat_out = np.degrees(lat3_rad)
    lon_out = (np.degrees(lon3_rad) + 540.0) % 360.0 - 180.0

    return lat_out, lon_out

def _point_is_on_semiretta(
    lat0: float, lon0: float, bearing_geo: float,
    lat_i: float, lon_i: float,
    tol_deg: float = 0.5,
) -> bool:
    """
    Verifica che il punto (lat_i, lon_i) si trovi sulla semiretta
    (non sul prolungamento opposto) definita da (lat0, lon0) con
    bearing geografico bearing_geo.

    Il test è: il bearing dal punto di origine verso il punto di intersezione
    deve essere entro tol_deg dal bearing atteso.
    """
    φ0, λ0 = np.radians(lat0), np.radians(lon0)
    φi, λi = np.radians(lat_i), np.radians(lon_i)
    Δλ = λi - λ0

    brg_to_i = np.degrees(
        np.arctan2(
            np.sin(Δλ) * np.cos(φi),
            np.cos(φ0) * np.sin(φi) - np.sin(φ0) * np.cos(φi) * np.cos(Δλ),
        )
    ) % 360.0

    diff = abs((brg_to_i - bearing_geo + 180.0) % 360.0 - 180.0)
    return diff <= tol_deg


def find_points(
    H1: tuple[float, float],
    H2: tuple[float, float],
    bearing_H1: list[float] | np.ndarray,
    bearing_H2: list[float] | np.ndarray,
) -> np.ndarray:
    """
    Calcola i punti di intersezione tra coppie di semirette geodetiche.

    Parametri
    ---------
    H1 : (lat, lon) del primo punto origine, in gradi decimali.
    H2 : (lat, lon) del secondo punto origine, in gradi decimali.
    bearing_H1 : array di N angoli (gradi) per H1,
                 misurati in senso ANTIORARIO a partire da EST.
    bearing_H2 : array di N angoli (gradi) per H2,
                 misurati in senso ANTIORARIO a partire da EST.

    Ritorna
    -------
    np.ndarray di forma (N, 2) con colonne [latitudine, longitudine].
    Le righe senza intersezione valida contengono [NaN, NaN].
    """
    bearing_H1 = np.asarray(bearing_H1, dtype=float)
    bearing_H2 = np.asarray(bearing_H2, dtype=float)

    if bearing_H1.shape != bearing_H2.shape:
        raise ValueError(
            f"bearing_H1 e bearing_H2 devono avere la stessa dimensione, "
            f"ma sono {bearing_H1.shape} e {bearing_H2.shape}."
        )

    n = bearing_H1.size
    result = np.full((n, 2), np.nan)

    lat1, lon1 = float(H1[0]), float(H1[1])
    lat2, lon2 = float(H2[0]), float(H2[1])

    for k in range(n):
        # Converti dalla convenzione matematica (CCW da EST)
        # al bearing geografico (CW da NORD)
        brg1 = math_to_bearing(bearing_H1[k])
        brg2 = math_to_bearing(bearing_H2[k])

        '''
        lat_i, lon_i = _great_circle_intersection(
            lat1, lon1, brg1,
            lat2, lon2, brg2,
        )
        '''

        lat_i, lon_i = _flat_earth_intersection(
            lat1, lon1, brg1,
            lat2, lon2, brg2,
        )

        if np.isnan(lat_i):
            continue  # nessuna intersezione geometrica

        # Verifica che il punto cada su entrambe le SEMI-rette
        # (non sul prolungamento opposto del raggio)
        on1 = _point_is_on_semiretta(lat1, lon1, brg1, lat_i, lon_i)
        on2 = _point_is_on_semiretta(lat2, lon2, brg2, lat_i, lon_i)

        if on1 and on2:
            result[k, 0] = lat_i
            result[k, 1] = lon_i
        # altrimenti rimane NaN

    return result

def find_points_3(
    H1: tuple[float, float],
    H2: tuple[float, float],
    H3: tuple[float, float],
    bearing_H1: list[float] | np.ndarray,
    bearing_H2: list[float] | np.ndarray,
    bearing_H3: list[float] | np.ndarray,
) -> np.ndarray:
    """
    Per ogni tripla di bearing, calcola il punto che minimizza la distanza
    dalle tre semirette geodetiche emesse da H1, H2, H3.

    Parametri
    ---------
    H1, H2, H3 : (lat, lon) dei tre idrofoni, in gradi decimali.
    bearing_H1/H2/H3 : array di N angoli (gradi), convenzione CCW da EST.

    Ritorna
    -------
    np.ndarray di forma (N, 2) con colonne [latitudine, longitudine].
    Le righe senza soluzione valida contengono [NaN, NaN].
    """
    bearing_H1 = np.asarray(bearing_H1, dtype=float)
    bearing_H2 = np.asarray(bearing_H2, dtype=float)
    bearing_H3 = np.asarray(bearing_H3, dtype=float)

    if not (bearing_H1.shape == bearing_H2.shape == bearing_H3.shape):
        raise ValueError(
            f"I tre array di bearing devono avere la stessa dimensione, "
            f"ma sono {bearing_H1.shape}, {bearing_H2.shape}, {bearing_H3.shape}."
        )

    n = bearing_H1.size
    result = np.full((n, 2), np.nan)

    lat1, lon1 = float(H1[0]), float(H1[1])
    lat2, lon2 = float(H2[0]), float(H2[1])
    lat3, lon3 = float(H3[0]), float(H3[1])

    for k in range(n):
        brg1 = math_to_bearing(bearing_H1[k])
        brg2 = math_to_bearing(bearing_H2[k])
        brg3 = math_to_bearing(bearing_H3[k])

        # Triangolo di errore
        candidates = []
        for (la, loa, ba), (lb, lob, bb) in [
            ((lat1, lon1, brg1), (lat2, lon2, brg2)),
            ((lat1, lon1, brg1), (lat3, lon3, brg3)),
            ((lat2, lon2, brg2), (lat3, lon3, brg3)),
        ]:
            lat_i, lon_i = _flat_earth_intersection(la, loa, ba, lb, lob, bb)
            if not np.isnan(lat_i):
                candidates.append((lat_i, lon_i))

        # RMS error
        opt = _least_squares_point(
            lat1, lon1, brg1,
            lat2, lon2, brg2,
            lat3, lon3, brg3,
        )

        if opt is None:
            if len(candidates) >= 2:
                lats = [p[0] for p in candidates]
                lons = [p[1] for p in candidates]
                opt = (float(np.mean(lats)), float(np.mean(lons)))
            else:
                continue

        lat_opt, lon_opt = opt
        result[k, 0] = lat_opt
        result[k, 1] = lon_opt
        
        '''
        # Accetta solo se il punto cade su tutte e tre le semirette
        on1 = _point_is_on_semiretta(lat1, lon1, brg1, lat_opt, lon_opt)
        on2 = _point_is_on_semiretta(lat2, lon2, brg2, lat_opt, lon_opt)
        on3 = _point_is_on_semiretta(lat3, lon3, brg3, lat_opt, lon_opt)

        
        if on1 and on2 and on3:
            result[k, 0] = lat_opt
            result[k, 1] = lon_opt
   
        checks = [on1, on2, on3]
        if sum(checks) >= 1:          # maggioranza
            result[k, 0] = lat_opt
            result[k, 1] = lon_opt
        '''
    return result


def _least_squares_point(
    lat1: float, lon1: float, brg1: float,
    lat2: float, lon2: float, brg2: float,
    lat3: float, lon3: float, brg3: float,
) -> tuple[float, float] | None:
    """
    Trova il punto (lat, lon) che minimizza la somma dei quadrati delle
    distanze dalle tre rette nel piano locale.

    Ogni retta è definita da un punto (latN, lonN) e un bearing brgN.
    La distanza di un punto P = (x, y) da una retta passante per
    Q = (qx, qy) con direzione unitaria d = (dx, dy) è:

        dist = | (P - Q) × d |  =  | (x-qx)·dy - (y-qy)·dx |

    Minimizzare la somma dei quadrati di queste distanze è un problema
    lineare ai minimi quadrati: A^T A x = A^T b, con soluzione analitica.
    """
    cos_lat = math.cos(math.radians((lat1 + lat2 + lat3) / 3))

    rows_a = []
    rows_b = []

    for lat, lon, brg in [(lat1, lon1, brg1), (lat2, lon2, brg2), (lat3, lon3, brg3)]:
        r = math.radians(brg)
        dx, dy = math.sin(r), math.cos(r)          # direzione unitaria

        qx = lon * cos_lat                          # piano locale scalato
        qy = lat

        # Coefficienti della distanza: dy·x - dx·y = dy·qx - dx·qy
        rows_a.append([dy, -dx])
        rows_b.append(dy * qx - dx * qy)

    A = np.array(rows_a)                            # (3, 2)
    b = np.array(rows_b)                            # (3,)

    # Soluzione ai minimi quadrati: (A^T A)^{-1} A^T b
    ATA = A.T @ A                                   # (2, 2)
    ATb = A.T @ b                                   # (2,)

    if abs(np.linalg.det(ATA)) < 1e-20:            # sistema degenere
        return None

    x_opt, y_opt = np.linalg.solve(ATA, ATb)

    lon_opt = x_opt / cos_lat
    lat_opt = y_opt

    return lat_opt, lon_opt


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def smooth_aoa(
    angles: np.ndarray,
    method: str = "savgol",
    window: int = 11,
    poly_order: int = 3,
    outlier_sigma: float = 2.5,
    wrap_degrees: bool = True,
) -> np.ndarray:
    """
    Smooths a time series of Angle of Arrival (AoA) measurements,
    reducing noise and suppressing outliers.
 
    The signal is assumed to be slowly varying (sampled at ~50ms intervals),
    with no abrupt direction changes expected.
 
    Parameters
    ----------
    angles : np.ndarray
        1-D array of AoA measurements in degrees.
    method : str
        Smoothing algorithm to apply after outlier removal:
          - "savgol"   : Savitzky-Golay filter (default). Best at preserving
                         trends while smoothing; ideal for slowly-varying signals.
          - "median"   : Running median. Very robust to bursts of outliers.
          - "gaussian" : Gaussian-weighted moving average. Maximum noise
                         suppression, slightly more lag.
    window : int
        Size of the smoothing window (number of samples). Must be odd for
        "savgol". Default is 11 (~550 ms at 50 ms/sample).
    poly_order : int
        Polynomial order for Savitzky-Golay (ignored for other methods).
        Must be < window. Default is 3.
    outlier_sigma : float
        Threshold (in standard deviations of the local residual) above which
        a sample is considered an outlier and replaced before smoothing.
        Higher values = less aggressive removal. Default is 2.5.
    wrap_degrees : bool
        If True the array is treated as a circular/angular quantity and
        unwrapped before processing, then re-wrapped to [0, 360) at the end.
        Set to False if your angles are already unwrapped or are in a
        non-circular range (e.g. elevation -90..+90). Default is True.
 
    Returns
    -------
    np.ndarray
        Smoothed angle array, same shape as input, in the same degree range
        as the input (wrapped to [0, 360) if wrap_degrees=True).
 
    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> t = np.linspace(0, 4 * np.pi, 200)
    >>> clean = 180 + 30 * np.sin(t)
    >>> noisy = clean + rng.normal(0, 3, size=clean.shape)
    >>> noisy[::20] += rng.choice([-25, 25], size=10)   # outliers
    >>> smoothed = smooth_aoa(noisy)
    """
    if angles.ndim != 1:
        raise ValueError("angles must be a 1-D array.")
 
    n = len(angles)
 
    # --- Ensure window is valid -------------------------------------------
    if window % 2 == 0:
        window += 1  # Savitzky-Golay requires odd window
    if window >= n:
        window = n if n % 2 != 0 else n - 1
 
    # --- 1. Unwrap circular angles ----------------------------------------
    if wrap_degrees:
        # np.unwrap works in radians
        work = np.unwrap(np.deg2rad(angles))
        work = np.rad2deg(work)
    else:
        work = angles.astype(float, copy=True)
 
    # --- 2. Outlier removal (iterative z-score on a running median) --------
    # Use a wide median filter as the "expected" signal, then flag samples
    # whose residual exceeds outlier_sigma standard deviations.
    med_win = min(window * 2 + 1, n if n % 2 != 0 else n - 1)
    baseline = median_filter(work, size=med_win, mode="nearest")
    residual = work - baseline
    sigma = np.std(residual)
 
    if sigma > 0:
        outlier_mask = np.abs(residual) > outlier_sigma * sigma
        # Replace outliers with the local median baseline value
        work = np.where(outlier_mask, baseline, work)
 
    # --- 3. Smoothing -------------------------------------------------------
    if method == "savgol":
        if poly_order >= window:
            poly_order = window - 1
        smoothed = savgol_filter(work, window_length=window, polyorder=poly_order,
                                 mode="nearest")
 
    elif method == "median":
        smoothed = median_filter(work, size=window, mode="nearest")
 
    elif method == "gaussian":
        # Build a Gaussian kernel
        sigma_g = window / 6.0  # 3-sigma covers half-window on each side
        half = window // 2
        x = np.arange(-half, half + 1)
        kernel = np.exp(-0.5 * (x / sigma_g) ** 2)
        kernel /= kernel.sum()
        smoothed = np.convolve(work, kernel, mode="same")
        # Fix edge effects with a reflected pad + reconvolve
        pad = np.pad(work, half, mode="reflect")
        smoothed = np.convolve(pad, kernel, mode="valid")
 
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'savgol', 'median', or 'gaussian'.")
 
    # --- 4. Re-wrap to [0, 360) if needed ----------------------------------
    if wrap_degrees:
        smoothed = smoothed % 360.0
 
    return smoothed



if __name__ == "__main__":
    with open("Synth/simulation_coordinates.txt", "r") as f:
        numer_of_floaters = f.readline()
        lat_tmp,lon_tmp = f.readline().split(' ')
        H1 = (float(lat_tmp),float(lon_tmp))
        lat_tmp,lon_tmp = f.readline().split(' ')
        H2 = (float(lat_tmp),float(lon_tmp))
        lat_tmp,lon_tmp = f.readline().split(' ')
        H3 = (float(lat_tmp),float(lon_tmp))

    bearing_H1 = np.load("Synth/H1.npy")
    bearing_H2 = np.load("Synth/H2.npy")
    bearing_H3 = np.load("Synth/H3.npy")
    bearing_H1 = bearing_H1[0:len(bearing_H1)-1]
    bearing_H2 = bearing_H2[0:len(bearing_H2)-1]
    bearing_H3 = bearing_H3[0:len(bearing_H3)-1]

    #intersections = find_points(H1, H2, bearing_H1, bearing_H2)
    intersection_points = find_points_3(H1,H2,H3,
                                  bearing_H1,
                                  bearing_H2,
                                  bearing_H3)
    np.save("Synth/Intersection_coordinates.npy",intersection_points)

    method = "gaussian" 
    #method = "savgol"    
    #method = "median"
    window = 15

    filt_bearing_H1 = smooth_aoa(
            bearing_H1,
            method=method,     # algoritmo
            window=window,           # 10 campioni = 550 ms di finestra
            poly_order=3,        # grado del polinomio locale
            outlier_sigma=3,   # soglia outlier (abbassa per essere più aggressivo)
            wrap_degrees=True,   # True se angoli in [0,360), False se già unwrappati
        )
    
    filt_bearing_H2 = smooth_aoa(
            bearing_H2,
            method=method,     # algoritmo
            window=window,           # 10 campioni = 550 ms di finestra
            poly_order=3,        # grado del polinomio locale
            outlier_sigma=3,   # soglia outlier (abbassa per essere più aggressivo)
            wrap_degrees=True,   # True se angoli in [0,360), False se già unwrappati
        )
    
    filt_bearing_H3 = smooth_aoa(
            bearing_H3,
            method=method,     # algoritmo
            window=window,           # 10 campioni = 550 ms di finestra
            poly_order=3,        # grado del polinomio locale
            outlier_sigma=3,   # soglia outlier (abbassa per essere più aggressivo)
            wrap_degrees=True,   # True se angoli in [0,360), False se già unwrappati
        )
    
    #filtered_intersections = find_points(H1, H2, filt_bearing_H1, filt_bearing_H2)
    intersection_points = find_points_3(H1,H2,H3,
                                  filt_bearing_H1,
                                  filt_bearing_H2,
                                  filt_bearing_H3)

    filt_bearing_H1 = filt_bearing_H1[0:len(filt_bearing_H1)-4]
    filt_bearing_H2 = filt_bearing_H2[0:len(filt_bearing_H2)-4]
    filt_bearing_H3 = filt_bearing_H3[0:len(filt_bearing_H3)-4]

    intersection_points = find_points_3(H1,H2,H3,
                                  filt_bearing_H1,
                                  filt_bearing_H2,
                                  filt_bearing_H3)
    
    print(bearing_H1)
    print()
    print(bearing_H2)
    print()
    print(bearing_H3)
    np.save("Synth/Filtered_Intersection_coordinates.npy",intersection_points)