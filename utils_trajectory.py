import numpy as np

def filter_trajectory_outliers(coordinates, window_size=5, max_distance=1000):
    """
    Rimuove gli outlier da un array di coordinate (Lat, Lon, [Depth])
    utilizzando un filtro a mediana mobile, con soglia espressa in metri.
    Non gestisce le interruzioni/salti di traiettoria (versione lineare).
    
    Parametri
    ---------
    coordinates : array-like di forma (K, 2) o (K, 3)
        Le coordinate della traiettoria da filtrare.
    window_size : int, opzionale (default=5)
        La dimensione della finestra mobile (deve essere un numero dispari).
    max_distance_meters : float, opzionale (default=200)
        La distanza massima consentita in metri tra il punto reale 
        e la mediana locale.
    """
    arr = np.asarray(coordinates, dtype=float)
    if len(arr) <= window_size:
        return arr  # Non ci sono abbastanza punti per filtrare
        
    lats = arr[:, 0]
    lons = arr[:, 1]
    
    # --- Convertitore Metri -> Gradi basato sulla latitudine media ---
    mean_lat = np.nanmean(lats)
    meters_per_degree_lat = 111132.95
    meters_per_degree_lon = 111412.84 * np.cos(np.radians(mean_lat))
    
    # Preallocazione array per le mediane
    median_lats = np.zeros_like(lats)
    median_lons = np.zeros_like(lons)
    
    half_w = window_size // 2
    
    # Calcolo della mediana mobile
    for i in range(len(arr)):
        start_idx = max(0, i - half_w)
        end_idx = min(len(arr), i + half_w + 1)
        
        median_lats[i] = np.nanmedian(lats[start_idx:end_idx])
        median_lons[i] = np.nanmedian(lons[start_idx:end_idx])
        
    # Calcolo dei delta rispetto alla mediana e conversione in metri
    delta_lat_m = (lats - median_lats) * meters_per_degree_lat
    delta_lon_m = (lons - median_lons) * meters_per_degree_lon
    
    # Distanza euclidea in metri
    distances_meters = np.sqrt(delta_lat_m**2 + delta_lon_m**2)
    
    # Filtraggio tramite maschera booleana
    valid_mask = distances_meters <= max_distance
    
    # Forza l'inclusione degli estremi della traiettoria
    valid_mask[0] = True
    valid_mask[-1] = True
    
    return arr[valid_mask]