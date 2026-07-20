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