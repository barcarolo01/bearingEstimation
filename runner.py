from datetime import datetime
import os
from discrete_hydromate import run_discrete_hydromate
from findpoint import find_points, filter_coordinates
from mappahtml import build_map
from utils_runner import *

N_steps = 3
N_H = 3

if not os.path.isdir("Synth"):
        os.makedirs("Synth")

# ==================== Coordinate calculation ====================
Lat_TX_init, Lon_TX_init = 20.832813, 88.698390 
Lat_TX_end, Lon_TX_end = sposta(Lat_TX_init, Lon_TX_init, 400, "E")

# Obtain the list of intermediate coordinates of the vessel
TX_Coordinates = compute_TX_trajectory(Lat_TX_init,Lon_TX_init,Lat_TX_end,Lon_TX_end,N_steps)
np.save("Synth/TX_Coordinates.npy",TX_Coordinates)

# Obtaining random coordinates for the floaters
middle_lat, middle_lon= TX_Coordinates[int((N_steps-1)/2),0], TX_Coordinates[int((N_steps-1)/2),1]
RX_Coordinates = random_points_within_distance(middle_lat,middle_lon,N_H,200,int(datetime.now().timestamp()))
np.save("Synth/RX_Coordinates.npy",RX_Coordinates)
# ================================================================

# Hydromate is launched from python: this produces three tracks for each floater of "Synth" folder
for i in range(N_H):
        run_discrete_hydromate(TX_Coordinates[:,0], TX_Coordinates[:,1],np.float64(RX_Coordinates[i,0]),np.float64(RX_Coordinates[i,1]),(i+1))

# Create the bearing angle array for each floater.
# This stores the bearing angle array of each floater in H{i}.npy.
first_bearing = compute_bearing_angle_array(1)
N_events = len(first_bearing)
bearing_arrays = np.zeros((N_H, N_events))
bearing_arrays[0,:] = first_bearing
for i in range(1, N_H):
    bearing_arrays[i,:] = compute_bearing_angle_array(i + 1)

# Deleting the temporary files
dir_path = "TMP"
for filename in os.listdir(dir_path):
    file_path = os.path.join(dir_path, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)
os.rmdir(dir_path)
print("TMP folder removed")

# Compute estimated points as the RMSE of the distance of the bearing directions
Estimated_positions = find_points(RX_Coordinates,bearing_arrays)

# Filtering the estimated points: remove all the points that appears less than twice
intersection_points = filter_coordinates(Estimated_positions,2)
np.save("Synth/Estimated_positions.npy",intersection_points)

# Plotting points on the map
build_map(
        floaters_coordinates = RX_Coordinates,
        TX_positions_coordinates = TX_Coordinates,
        estimated_vessel_coordinates = Estimated_positions,
        output_file="map.html"
    )