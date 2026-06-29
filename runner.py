import time
import os
from dotenv import load_dotenv
from coordinate_generator import compute_TX_circle_trajectory, sposta
from discrete_hydromate import run_discrete_hydromate
from filtratraiettoria import *
from findpoint import *
from build_folium_map import build_map
from build_local_map import build_local_cartesian_map
from utils_runner import *

SIMULATE = True

load_dotenv()
NUMBER_OF_HYDROPHONES = int(os.getenv('NUMBER_OF_HYDROPHONES'))
SAMPLING_FREQUENCY = int(os.getenv('SAMPLING_FREQUENCY'))

if not os.path.isdir("Synth"):
        os.makedirs("Synth")

# ========== Coordinate generation ==========
Lat_center, Lon_center = 20.832813, 88.698390
TX_Coordinates = compute_TX_circle_trajectory(Lat_center,Lon_center,10,90,270,10,200,True)
np.save("Synth/TX_Coordinates.npy",TX_Coordinates)

lat1,lon1 = sposta(Lat_center,Lon_center,100,90)
lat2,lon2 = sposta(Lat_center,Lon_center,100,270)
RX_Coordinates = np.asarray([[lat1,lon1],[lat2,lon2]])
np.save("Synth/RX_Coordinates.npy",RX_Coordinates)
NUMBER_OF_FLOATERS = RX_Coordinates.shape[0]


if TX_Coordinates.shape[1] == 2:
    filler = np.ones(TX_Coordinates.shape[0]).reshape(-1, 1) * 10
    TX_Coordinates = np.hstack((TX_Coordinates, filler))
    print(f"Transmitter array do not have a depth value. Filled with default value of of 10 meters")

if RX_Coordinates.shape[1] == 2:
    filler = np.ones(RX_Coordinates.shape[0]).reshape(-1, 1) * 10
    RX_Coordinates = np.hstack((RX_Coordinates, filler))
    print("Floaters array do not have a depth value. Filled with default value of 10 meters")

        
# Hydromate is launched from python: this produces three tracks for each floater of "Synth" folder
if SIMULATE:
        for i in range(NUMBER_OF_FLOATERS):
                print(f" ## Floater number {i+1} ##")
                run_discrete_hydromate(TX_Coordinates[:,0],
                                        TX_Coordinates[:,1],
                                        TX_Coordinates[:,2],
                                        np.float64(RX_Coordinates[i,0]),
                                        np.float64(RX_Coordinates[i,1]),
                                        np.float64(RX_Coordinates[i,2]),
                                        (i+1))
                
                # Write for each hydrophone the .wav track generated
                for j in range(NUMBER_OF_HYDROPHONES):
                        array = np.load(f'Synth/F{i+1}_H{j+1}.npy')
                        wav.write(f'Synth/F{i+1}_H{j+1}.wav', SAMPLING_FREQUENCY, array)
                
                time.sleep(0.5)


# Create the bearing angle array for each floater: this stores the bearing angle array of each floater in H{i}.npy.
if NUMBER_OF_HYDROPHONES == 3:
        first_bearing = compute_bearing_angle_array(1)
        elevation_arrays = np.zeros((NUMBER_OF_FLOATERS,len(first_bearing)))   
elif NUMBER_OF_HYDROPHONES == 4:
        first_bearing = compute_bearing_angle_array_square(1)
        elevation_arrays = np.zeros((NUMBER_OF_FLOATERS,len(first_bearing)))   
else:
        first_bearing,first_elevation = compute_bearing_angle_array_complete(1)

# Creating bearing and elevation arrays on the base of the number of event previously fetched
N_events = len(first_bearing)
bearing_arrays = np.zeros((NUMBER_OF_FLOATERS, N_events))
elevation_arrays = np.full((NUMBER_OF_FLOATERS,N_events),np.nan)
bearing_arrays[0,:] = first_bearing

if NUMBER_OF_HYDROPHONES == 5:
        elevation_arrays[0,:] = first_elevation

for i in range(1, NUMBER_OF_FLOATERS):
        if NUMBER_OF_HYDROPHONES == 3:
                bearing_arrays[i,:] = compute_bearing_angle_array(i + 1)
        elif NUMBER_OF_HYDROPHONES == 4:
                bearing_arrays[i,:] = compute_bearing_angle_array_square(i + 1)
        else:
                bearing_arrays[i,:], elevation_arrays[i,:] = compute_bearing_angle_array_complete(i + 1)
        
# Deleting the temporary files and folder
clean_temporary_files('TMP')

intersection_points = find_points(RX_Coordinates,bearing_arrays,elevation_arrays)
#intersection_points = find_points_weighted(RX_Coordinates,bearing_arrays,elevation_arrays)

intersection_points = group_close_points(intersection_points,tolleranza_metri=2)
intersection_points = remove_outliers_median(intersection_points,9)

np.save("Synth/Estimated_Coordinates",intersection_points)
# Plotting points on the map
build_map(
        floaters_coordinates = RX_Coordinates,
        TX_positions_coordinates = TX_Coordinates,
        estimated_vessel_coordinates = intersection_points,
        output_file="map_folium.html",
        track_TX = True,
        track_estimated = True
    )


center = np.load("Synth/Center_Coordinates.npy")

# Finestra di 1000 metri di larghezza e 800 metri di altezza
build_local_cartesian_map(
RX_Coordinates, 
TX_Coordinates, 
intersection_points, 
center_coordinates=center,
window_width_m=1000, 
window_height_m=1000,
output_file="map_local.png",
track_TX=True,
track_estimated=True
)

rmse = compute_geometric_rmse(TX_Coordinates,intersection_points)
print(f"Geometric RMSE: {rmse:1f}")
print(f"Average estimated TX depth: {np.average(intersection_points[:,2]):1f}")