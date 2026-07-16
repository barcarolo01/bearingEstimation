import numpy as np
import pyproj
from scipy.ndimage import median_filter



def remove_outliers_median(coordinate, WIN_LEN=7):
    '''
    Removes outliers by applying a median filtering of specified WIN_LEN to latitude,
    longitude and depth (if available).
    
    Inputs:
        coordinate: numpy array of size (N,2) or (N,3)
        WIN_LEN: length of the median filter window
    Output:
        filtered_coordinates: numpy array of filited coordinates with shape of 'coordinate'
    '''
    filtered_coordinates = np.zeros_like(coordinate)
    for i in range(coordinate.shape[1]):
        filtered_coordinates[:, i] = median_filter(coordinate[:, i], size=WIN_LEN, mode='nearest')
        
    return filtered_coordinates

def replace_outliers_mean(coordinate, WIN_LEN=7):
    '''
    Replaces outliers/points by substituting them with the mean of the M previous 
    and M subsequent samples (excluding the central point itself), ignorando i NaN
    presenti nella finestra invece di propagarli.
    '''
    M = WIN_LEN // 2
    
    kernel = np.ones(WIN_LEN)
    kernel[M] = 0  # esclude il campione centrale
    
    filtered_coordinates = np.full_like(coordinate, np.nan, dtype=float)
    
    for i in range(coordinate.shape[1]):
        col = coordinate[:, i].astype(float)
        padded_col = np.pad(col, M, mode='edge')
        
        valid_mask = ~np.isnan(padded_col)
        padded_filled = np.where(valid_mask, padded_col, 0.0)
        
        # somma pesata dei soli valori validi nella finestra
        weighted_sum = np.convolve(padded_filled, kernel, mode='valid')
        # numero di valori validi pesati nella finestra (denominatore "adattivo")
        weighted_count = np.convolve(valid_mask.astype(float), kernel, mode='valid')
        
        with np.errstate(invalid='ignore', divide='ignore'):
            filtered_coordinates[:, i] = weighted_sum / weighted_count
    
    return filtered_coordinates




def group_close_points(dati, tolleranza_metri=2.0):
    '''
    Groups consecutive points that are closer than 'tolerance_meter',
    substituting them with their geometric mean.
    '''

    if len(dati) == 0:
        return dati
        
    METRI_PER_GRADO_LAT = 111320.0
    METRI_PER_GRADO_LON = 104600.0
    
    punti_filtrati = []
    cluster_corrente = [dati[0]]
    
    for i in range(1, len(dati)):
        # Calcola distanza tra il punto corrente e l'inizio del cluster
        diff_lat = (dati[i, 0] - cluster_corrente[0][0]) * METRI_PER_GRADO_LAT
        diff_lon = (dati[i, 1] - cluster_corrente[0][1]) * METRI_PER_GRADO_LON
        distanza = np.sqrt(diff_lat**2 + diff_lon**2)
        
        if distanza <= tolleranza_metri:
            # Il punto è molto vicino, lo aggiungo al gruppo per fare la media
            cluster_corrente.append(dati[i])
        else:
            # Il punto si è allontanato: chiudo il cluster precedente facendo la media
            punti_filtrati.append(np.mean(cluster_corrente, axis=0))
            # Faccio partire un nuovo cluster con il punto attuale
            cluster_corrente = [dati[i]]
            
    # Non dimentichiamo l'ultimo cluster rimasto aperto
    punti_filtrati.append(np.mean(cluster_corrente, axis=0))
    
    return np.array(punti_filtrati)


def compute_RMSE_same_size(st, gt, lat_rad, lon_rad):
    METRI_PER_GRADO_LAT = (
        111132.92
        - 559.82 * np.cos(2 * lat_rad)
        + 1.175 * np.cos(4 * lat_rad)
        - 0.0023 * np.cos(6 * lat_rad)
    )
    METRI_PER_GRADO_LON = (
        111412.84 * np.cos(lat_rad)
        - 93.5 * np.cos(3 * lat_rad)
        + 0.118 * np.cos(5 * lat_rad)
    )

    # maschera comune sui NaN, per mantenere l'allineamento point-wise
    mask = ~(np.isnan(gt).any(axis=1) | np.isnan(st).any(axis=1))
    gt = gt[mask]
    st = st[mask]

    if gt.shape[0] == 0:
        return np.nan  # nessun punto valido in comune

    gt_m = gt * [METRI_PER_GRADO_LAT, METRI_PER_GRADO_LON]
    st_m = st * [METRI_PER_GRADO_LAT, METRI_PER_GRADO_LON]

    diff = gt_m - st_m
    dist2 = np.sum(diff**2, axis=1)   # distanza euclidea al quadrato, per ogni punto

    rmse = np.sqrt(np.mean(dist2))    # un solo numero, in metri
    return rmse

def compute_depth_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return np.nan

    diff = y_true[mask] - y_pred[mask]
    return np.sqrt(np.mean(diff ** 2))













def clean_trajectory_3d(
    lat: np.ndarray,
    lon: np.ndarray,
    depth: np.ndarray,
    window_size: int = 5,
    threshold_xy_meters: float = 80.0,
    threshold_z_meters: float = 5.0,
):
    """Filtra gli outlier da traiettorie 3D (Lat, Lon, Depth) usando una mediana mobile

    e sostituendo i picchi anomali con il valore mediano locale.
    """
    # 1. CONVERSIONE IN METRI (WGS84 -> UTM)
    # Identifichiamo la zona UTM ottimale partendo dal centro del dataset
    mean_lat, mean_lon = np.nanmean(lat), np.nanmean(lon)
    utm_zone = int((mean_lon + 180) / 6) + 1
    hemisphere = "north" if mean_lat >= 0 else "south"

    # Inizializziamo il transformer (reversibile)
    proj_wgs84 = pyproj.CRS("EPSG:4326")
    proj_utm = pyproj.CRS(
        f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"
    )
    transformer_to_meters = pyproj.Transformer.from_crs(
        proj_wgs84, proj_utm, always_xy=True
    )
    transformer_to_deg = pyproj.Transformer.from_crs(
        proj_utm, proj_wgs84, always_xy=True
    )

    # Convertiamo Lat/Lon in X/Y metrici
    x, y = transformer_to_meters.transform(lon, lat)
    x = np.array(x)
    y = np.array(y)
    z = np.copy(depth)  # La profondità è già in metri

    # 2. CALCOLO DELLA MEDIANA MOBILE (La nostra "combinazione dei vicini")
    # Il median_filter di scipy mantiene la stessa lunghezza dell'array
    x_med = median_filter(x, size=window_size, mode="nearest")
    y_med = median_filter(y, size=window_size, mode="nearest")
    z_med = median_filter(z, size=window_size, mode="nearest")

    # 3. IDENTIFICAZIONE E SOSTITUZIONE OUTLIER
    # Calcoliamo la distanza euclidea planare (2D) dalla mediana
    distance_xy = np.sqrt((x - x_med) ** 2 + (y - y_med) ** 2)
    outliers_xy = distance_xy > threshold_xy_meters

    # Calcoliamo lo scostamento sulla profondità (1D)
    distance_z = np.abs(z - z_med)
    outliers_z = distance_z > threshold_z_meters

    # Sostituiamo i valori anomali con il rispettivo valore mediano
    x_cleaned = np.where(outliers_xy, x_med, x)
    y_cleaned = np.where(outliers_xy, y_med, y)
    z_cleaned = np.where(outliers_z, z_med, z)

    # 4. RICONVERSIONE IN GRADI
    lon_cleaned, lat_cleaned = transformer_to_deg.transform(
        x_cleaned, y_cleaned
    )

    return np.array(lat_cleaned), np.array(lon_cleaned), np.array(z_cleaned)