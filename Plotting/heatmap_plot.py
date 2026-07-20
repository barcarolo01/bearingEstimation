import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import uniform_filter
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

FONT_SIZE = 18

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE
})

# ========== PARAMETERS ==========
CSV_FILE = "Plotting/grid_errors_for_heatmap.csv"  

B = 200     # Baseline (distance between the flaoters, in meters)

USE_DB = True         # If True, the error is visualized in dB [10*log10(Error)], otherwise in meters
APPLY_FILTER = True     # If True, a uniform filter is applied to sampled data on the grid before visualization
FILTER_SIZE = 5

FILL_NAN_WITH_MAX = False   # If True, NaNs values are substituted with the maximum finite error value
COLORMAP = 'plasma'
INTERPOLATION_METHOD = 'linear'


# Local origin coordinates
LAT_ORIGIN = 20.832813
LON_ORIGIN = 88.698390
R_EARTH = 6378137.0  # Earth's radius in meters

def geodetic_to_local_meters(lat, lon, lat_org, lon_org):
    """Converts Lat/Lon coordinates to local meters (X=East, Y=Nord)"""
    y = (lat - lat_org) * (np.pi / 180.0) * R_EARTH
    factor_lon = R_EARTH * np.cos(np.radians(lat_org))
    x = (lon - lon_org) * (np.pi / 180.0) * factor_lon
    return x, y


# DATA LOADING AND PREPARATION
df = pd.read_csv(CSV_FILE)

# Converts Ground Truth (GT) coordinates to local meters
df['X_GT'], df['Y_GT'] = geodetic_to_local_meters(df['Lat_GT'], df['Lon_GT'], LAT_ORIGIN, LON_ORIGIN)

# Reads the RMSE column from the CSV
df['Error_m'] = df['RMSE']

# Conditional handling of NaNs before interpolation
if FILL_NAN_WITH_MAX:
    max_real_error = df['Error_m'].max()
    df['Error_m'] = df['Error_m'].fillna(max_real_error)
else:
    df_interp = df.dropna(subset=['Error_m'])

points_df = df if FILL_NAN_WITH_MAX else df_interp


# REGULAR GRID INTERPOLATION
x_min, x_max = df['X_GT'].min(), df['X_GT'].max()
y_min, y_max = df['Y_GT'].min(), df['Y_GT'].max()

grid_x, grid_y = np.mgrid[x_min:x_max:300j, y_min:y_max:300j]

grid_error = griddata(
    points=(points_df['X_GT'].values, points_df['Y_GT'].values),
    values=points_df['Error_m'].values,
    xi=(grid_x, grid_y),
    method=INTERPOLATION_METHOD
)


# OPTIONAL PROCESSING (FILTER AND SCALE)
if APPLY_FILTER:
    if not FILL_NAN_WITH_MAX and np.isnan(grid_error).any():
        nan_mask = np.isnan(grid_error)
        grid_error_filled = np.where(nan_mask, 0, grid_error)
        counts = uniform_filter(np.logical_not(nan_mask).astype(float), size=FILTER_SIZE, mode='reflect')
        grid_error = uniform_filter(grid_error_filled, size=FILTER_SIZE, mode='reflect') / np.where(counts == 0, 1, counts)
        grid_error[nan_mask] = np.nan
    else:
        grid_error = uniform_filter(grid_error, size=FILTER_SIZE, mode='reflect')

if USE_DB:
    grid_error_plot = 10 * np.log10(np.clip(grid_error, 0.1, None))
    cbar_label = 'Localization error [dB]'
    norm_color = Normalize(vmin=np.nanmin(grid_error_plot), vmax=np.nanmax(grid_error_plot))
else:
    grid_error_plot = grid_error
    cbar_label = 'Localization error [meters]'
    norm_color = LogNorm(vmin=max(0.1, np.nanmin(grid_error_plot)), vmax=np.nanmax(grid_error_plot))
    print(f"MIN = {np.nanmin(grid_error_plot)} MAX = {np.nanmax(grid_error_plot)}")


# HEATMAP PLOT
fig, ax = plt.subplots(figsize=(13, 6))

heatmap = ax.imshow(
    grid_error_plot.T, 
    extent=(x_min, x_max, y_min, y_max), 
    origin='lower',
    cmap=COLORMAP, 
    norm=norm_color
)


divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad="5%")
cbar = fig.colorbar(heatmap, cax=cax)
cbar.set_label(cbar_label, labelpad=15, size=FONT_SIZE)
cax.tick_params(labelsize=FONT_SIZE)

# Highlight the two sensors on the main axis 'ax'
sensor1_x, sensor1_y = -B/2, 0
sensor2_x, sensor2_y = B/2, 0
ax.scatter(sensor1_x, sensor1_y, color='red', marker='s', s=150, edgecolor='black', zorder=5)
ax.scatter(sensor2_x, sensor2_y, color='red', marker='s', s=150, edgecolor='black', zorder=5)

ax.set_xlabel('West-East [meters]')
ax.set_ylabel('North distance [meters]')

plt.tight_layout()
plt.savefig("FiguresAndPlots/localization_error_heatmap.eps")
plt.show()