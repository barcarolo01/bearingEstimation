import numpy as np
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

def compute_RMSE(gt, est):
    """
    Computes the planar and depth RMSE error between two sequences of cooordinates

    Parameters
    ---------
    gt, est : numpy arrays of shape (N, 2) o (N, 3)
        Columns: [latitude (degrees), longitude (degrees), depth (meters, optional)]

    Returns
    -----------
    rmse_flat : float
        RMSE of the planar point-to-point distance, expressed in meters
    rmse_depth : float
        RMSE of the depth point-to-point distance, expressed in meters (or np.nan is input arrays have shape(N,2))
    """

    METERS_PER_DEG = 111_319.9
    gt = np.asarray(gt, dtype=float)
    est = np.asarray(est, dtype=float)

    if gt.shape != est.shape:
        raise ValueError(f"Input arrays have different shapes: gt {gt.shape}, est {est.shape}")
    if gt.ndim != 2 or gt.shape[1] not in (2, 3):
        raise ValueError(f"Expected shape (N, 2) or (N, 3), while received {gt.shape}")

    # === Planar RMSE  ===
    gt_ll, est_ll = gt[:, :2], est[:, :2]
    mask_flat = ~(np.isnan(gt_ll).any(axis=1) | np.isnan(est_ll).any(axis=1))

    if np.any(mask_flat):
        diff = gt_ll[mask_flat] - est_ll[mask_flat]
        diff[:, 0] *= METERS_PER_DEG
        diff[:, 1] *= METERS_PER_DEG * np.cos(np.deg2rad(gt_ll[mask_flat, 0]))
        rmse_flat = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    else:
        rmse_flat = np.nan

    # === Depth RMSE ===
    rmse_depth = np.nan
    if gt.shape[1] == 3:
        d_gt, d_est = gt[:, 2], est[:, 2]
        mask_depth = ~(np.isnan(d_gt) | np.isnan(d_est))
        if np.any(mask_depth):
            diff_d = d_gt[mask_depth] - d_est[mask_depth]
            rmse_depth = np.sqrt(np.mean(diff_d**2))

    return rmse_flat, rmse_depth