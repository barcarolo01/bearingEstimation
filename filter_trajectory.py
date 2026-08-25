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

'''
Computes the RMSE error only along the depth coordinate of a series of 3D points
'''
def compute_flat_RMSE(st, gt, lat_rad_center, lon_rad_center):
    # Generate a mask for NaN values of both sequences (to maintain the point-wise alignment)
    mask = ~(np.isnan(gt).any(axis=1) | np.isnan(st).any(axis=1))

    # Coordinate differences, in degrees
    diff = gt[mask] - st[mask]

    # Distance conversion in meters
    diff[:,0] *= 111_319.9
    diff[:,1] *= (111_319.9 * np.cos(np.deg2rad((gt[:,0]))))

    # Return a single number: average RMSE of the distances
    rmse = np.sqrt(np.sum(diff**2))
    return np.mean(rmse)

'''
Computes the RMSE error only along the depth coordinate of a series of 3D points
'''
def compute_depth_RMSE(depth_true, depth_pred):
    depth_true = np.asarray(depth_true, dtype=float)
    depth_est = np.asarray(depth_pred, dtype=float)

    mask = ~np.isnan(depth_true) & ~np.isnan(depth_est)
    if not np.any(mask):
        return np.nan

    diff = depth_true[mask] - depth_est[mask]
    rmse = np.sqrt(np.sum(diff**2))
    return np.mean(rmse)