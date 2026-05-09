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

def find_points(
    hydrophones: np.ndarray,
    bearings: np.ndarray,
) -> np.ndarray:
    """
    Per ogni colonna di bearing, calcola il punto che minimizza la distanza
    dalle N semirette geodetiche emesse dagli N idrofoni.

    Parametri
    ---------
    hydrophones : np.ndarray di forma (N, 2)
        Coordinate [lat, lon] in gradi decimali di N idrofoni.
    bearings : np.ndarray di forma (N, M)
        M set di N angoli (gradi), convenzione CCW da EST.
        Ogni colonna m contiene i bearing dei N idrofoni per l'evento m.

    Ritorna
    -------
    np.ndarray di forma (M, 2) con colonne [latitudine, longitudine].
    Le righe senza soluzione valida contengono [NaN, NaN].
    """
    hydrophones = np.asarray(hydrophones, dtype=float)
    bearings = np.asarray(bearings, dtype=float)

    if hydrophones.ndim != 2 or hydrophones.shape[1] != 2:
        raise ValueError(
            f"hydrophones deve avere forma (N, 2), ma ha forma {hydrophones.shape}."
        )
    if bearings.ndim != 2:
        raise ValueError(
            f"bearings deve avere forma (N, M), ma ha forma {bearings.shape}."
        )

    n_hydro = hydrophones.shape[0]
    if bearings.shape[0] != n_hydro:
        raise ValueError(
            f"bearings deve avere {n_hydro} righe (una per idrofono), "
            f"ma ne ha {bearings.shape[0]}."
        )
    if n_hydro < 2:
        raise ValueError("Servono almeno 2 idrofoni.")

    n_events = bearings.shape[1]
    result = np.full((n_events, 2), np.nan)

    # Pre-converti tutti i bearing in convenzione geografica
    brgs = np.vectorize(math_to_bearing)(bearings)  # (N, M)

    lats = hydrophones[:, 0]
    lons = hydrophones[:, 1]

    for m in range(n_events):
        brg_m = brgs[:, m]  # (N,) bearing per l'evento m

        if n_hydro == 2:
            # Caso esatto: intersezione di due rette
            lat_i, lon_i = _flat_earth_intersection(
                lats[0], lons[0], brg_m[0],
                lats[1], lons[1], brg_m[1],
            )
            if not np.isnan(lat_i):
                result[m, 0] = lat_i
                result[m, 1] = lon_i
            continue

        # Caso N >= 3: triangolo/poligono di errore + least squares
        # Raccogli tutti i vertici del poligono di errore (intersezioni a coppie)
        candidates = []
        for i in range(n_hydro):
            for j in range(i + 1, n_hydro):
                lat_i, lon_i = _flat_earth_intersection(
                    lats[i], lons[i], brg_m[i],
                    lats[j], lons[j], brg_m[j],
                )
                if not np.isnan(lat_i):
                    candidates.append((lat_i, lon_i))

        # Least squares generalizzato su tutti gli N idrofoni
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

        result[m, 0] = opt[0]
        result[m, 1] = opt[1]

    return result

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

def filter_coordinates(coords: np.ndarray, f: int) -> np.ndarray:
    """
    Rimuove da un array di coordinate (LAT, LON) tutte le coppie
    che compaiono meno di F volte.
 
    Parametri
    ----------
    coords : np.ndarray, shape (K, 2)
        Array di coordinate dove la colonna 0 è la latitudine
        e la colonna 1 è la longitudine.
    f : int
        Frequenza minima: vengono mantenute solo le coppie (LAT, LON)
        che compaiono almeno F volte.
 
    Ritorna
    -------
    np.ndarray, shape (M, 2)
        Array filtrato contenente solo le righe la cui coppia
        (LAT, LON) compare almeno F volte nell'input.
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"L'array deve avere forma (K, 2), ricevuto: {coords.shape}")
    if f < 1:
        raise ValueError(f"La frequenza minima F deve essere >= 1, ricevuto: {f}")
 
    # Converte ogni riga in una tupla hashable per contarle
    tuples = [tuple(row) for row in coords]
    counts = Counter(tuples)
 
    # Maschera booleana: True dove la coppia compare almeno F volte
    mask = np.array([counts[t] >= f for t in tuples])
 
    return coords[mask]

if __name__ == "__main__":
    RX_Coordinates = np.load("Synth/RX_Coordinates.npy")
    '''
    bearing_H1 = np.load("Synth/H1.npy")
    bearing_H2 = np.load("Synth/H2.npy")
    bearing_H3 = np.load("Synth/H3.npy")
    bearing_H4 = np.load("Synth/H4.npy")
    bearing_H5 = np.load("Synth/H5.npy")

    bearing_arrays = np.asarray([bearing_H1,bearing_H2,bearing_H3,bearing_H4,bearing_H5])
    intersection_points = find_points(RX_Coordinates,bearing_arrays)
        
    np.save("Synth/Intersection_coordinates.npy",intersection_points)
    '''