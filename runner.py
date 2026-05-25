import time
import os
from dotenv import load_dotenv
from discrete_hydromate import run_discrete_hydromate
from findpoint import *
from build_map import build_map
from utils_runner import *

SIMULATE = True

load_dotenv()
NUMBER_OF_STEPS = int(os.getenv('NUMBER_OF_STEPS'))
NUMBER_OF_FLOATERS = int(os.getenv('NUMBER_OF_FLOATERS'))
NUMBER_OF_HYDROPHONES = int(os.getenv('NUMBER_OF_HYDROPHONES'))
SAMPLING_FREQUENCY = int(os.getenv('SAMPLING_FREQUENCY'))

if not os.path.isdir("Synth"):
        os.makedirs("Synth")

# ==================== Coordinate calculation ====================
constant_TX_depth = 10.0
constant_RX_depth = 10.0

Lat_TX_init, Lon_TX_init = 20.832813, 88.698390 
#Lat_TX_init,Lon_TX_init = -35.9570,  153.3208 
Lat_TX_end, Lon_TX_end = sposta(Lat_TX_init, Lon_TX_init, 1000, 90)

# Obtain the list of intermediate coordinates of the vessel
TX_Coordinates = compute_TX_trajectory(Lat_TX_init,Lon_TX_init,Lat_TX_end,Lon_TX_end,constant_TX_depth,NUMBER_OF_STEPS)

#np.save("Synth/TX_Coordinates.npy",TX_Coordinates)
#TX_Coordinates = np.load("Synth/TX_Coordinates.npy")
#TX_Coordinates = TX_Coordinates[570:570+NUMBER_OF_STEPS,:].astype(np.float64)


# Obtaining coordinates of the floaters
'''
middle_lat = TX_Coordinates[int(NUMBER_OF_STEPS/2),0]
middle_lon = TX_Coordinates[int(NUMBER_OF_STEPS/2),1]
RX_Coordinates = random_points_within_distance_recursive(middle_lat,
                                                         middle_lon,
                                                         constant_RX_depth,
                                                         NUMBER_OF_FLOATERS,
                                                         800,
                                                         int(120))

                                                         '''

lat1,lon1 = sposta(TX_Coordinates[0,0],TX_Coordinates[0,1],400,0)
lat2,lon2 = sposta(TX_Coordinates[1,0],TX_Coordinates[1,1],400,180)
lat3,lon3 = sposta(TX_Coordinates[2,0],TX_Coordinates[2,1],400,0)
lat4,lon4 = sposta(TX_Coordinates[3,0],TX_Coordinates[3,1],400,180)
lat5,lon5 = sposta(TX_Coordinates[4,0],TX_Coordinates[4,1],400,0)

RX_Coordinates = np.asarray([[lat1,lon1],[lat2,lon2],[lat3,lon3],[lat4,lon4],[lat5,lon5]])

np.save("Synth/TX_Coordinates.npy",TX_Coordinates)
np.save("Synth/RX_Coordinates.npy",RX_Coordinates)

if TX_Coordinates.shape[1] == 2:
    filler = np.ones(TX_Coordinates.shape[0]).reshape(-1, 1) * constant_TX_depth
    TX_Coordinates = np.hstack((TX_Coordinates, filler))

if RX_Coordinates.shape[1] == 2:
    filler = np.ones(RX_Coordinates.shape[0]).reshape(-1, 1) * constant_RX_depth
    RX_Coordinates = np.hstack((RX_Coordinates, filler))

print("?"*20)
print(TX_Coordinates)

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
                
                for j in range(NUMBER_OF_HYDROPHONES):
                        array = np.load(f'Synth/F{i+1}_H{j+1}.npy')
                        wav.write(f'Synth/F{i+1}_H{j+1}.wav', SAMPLING_FREQUENCY, array)
                
                time.sleep(1)


# Create the bearing angle array for each floater.
# This stores the bearing angle array of each floater in H{i}.npy.

if NUMBER_OF_HYDROPHONES == 3:
        first_bearing = compute_bearing_angle_array(1)
        elevation_arrays = np.zeros((NUMBER_OF_FLOATERS,len(first_bearing)))   
elif NUMBER_OF_HYDROPHONES == 4:
        first_bearing = compute_bearing_angle_array_square(1)
        elevation_arrays = np.zeros((NUMBER_OF_FLOATERS,len(first_bearing)))   
else:
        first_bearing,first_elevation = compute_bearing_angle_array_complete(1)

# Creating bearing and elevation arrays
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

# Plotting points on the map
build_map(
        floaters_coordinates = RX_Coordinates,
        TX_positions_coordinates = TX_Coordinates,
        estimated_vessel_coordinates = intersection_points,
        output_file="map.html"
    )

print()
print()