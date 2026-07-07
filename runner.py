import os
from dotenv import load_dotenv
from build_local_3D import build_local_cartesian_map_3d
from coordinate_generator import compute_TX_circle_trajectory, sposta
from discrete_hydromate import run_discrete_hydromate
from filtratraiettoria import *
from findpoint import *
from build_folium_map import build_map
from build_local_map import build_local_cartesian_map
from utils_runner import *

SIMULATE = True
ANALYZE_WAVS = False

load_dotenv()
NUMBER_OF_HYDROPHONES = int(os.getenv('NUMBER_OF_HYDROPHONES'))
SAMPLING_FREQUENCY = int(os.getenv('SAMPLING_FREQUENCY'))

if not os.path.isdir("Synth"):
        os.makedirs("Synth")

# ========== Coordinate generation ==========
Lat_center, Lon_center = 20.832813, 88.698390
d_RX1 = 10
d_RX2 = 10
d_RX3 = 10
d_TX = 50
#TX_Coordinates = np.asarray([[Lat_center,Lon_center,d_TX]])
TX_Coordinates = compute_TX_circle_trajectory(Lat_center, Lon_center,d_TX,0,350,4,200,clockwise=True)

'''
depths = np.zeros(TX_Coordinates.shape[0])
depth_init = 90
depth_end = 10

depths[0] = depth_init
for i in range(1,depths.shape[0]):
        depths[i] = depth_init - i*(np.abs(depth_end-depth_init)/(depths.shape[0]-1))

TX_Coordinates[:,2] = depths
'''

lat1,lon1 = sposta(Lat_center,Lon_center,115,0)
lat2,lon2 = sposta(Lat_center,Lon_center,115,120)
lat3,lon3 = sposta(Lat_center,Lon_center,115,240)
RX_Coordinates = np.asarray([[lat1,lon1,d_RX1],[lat2,lon2,d_RX2],[lat3,lon3,d_RX3]])
NUMBER_OF_FLOATERS = RX_Coordinates.shape[0]

np.save("Synth/TX_Coordinates.npy",TX_Coordinates)
np.save("Synth/Center_Coordinates.npy",[Lat_center,Lon_center])
np.save("Synth/RX_Coordinates.npy",RX_Coordinates)

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
                
            

if SIMULATE or ANALYZE_WAVS:
        # Create the bearing angle array for each floater: this stores the bearing angle array of each floater in H{i}.npy.
        if NUMBER_OF_HYDROPHONES == 3:
                first_bearing = compute_bearing_angle_array(1)
                elevation_arrays = np.zeros((NUMBER_OF_FLOATERS,len(first_bearing)))   
        elif NUMBER_OF_HYDROPHONES == 4:
                first_bearing = compute_bearing_angle_array_square(1)
                elevation_arrays = np.zeros((NUMBER_OF_FLOATERS,len(first_bearing)))   
        else:
                first_bearing,first_elevation = compute_bearing_angle_array_complete(1)
                np.save(f"Synth/F1_elevation.npy",first_elevation)

        # Creating bearing and elevation arrays on the base of the number of event previously fetched
        N_events = len(first_bearing)
        bearing_arrays = np.zeros((NUMBER_OF_FLOATERS, N_events))
        elevation_arrays = np.full((NUMBER_OF_FLOATERS,N_events),np.nan)
        bearing_arrays[0,:] = first_bearing
        np.save(f"Synth/F1_azimuth.npy",first_bearing)
        
 
        if NUMBER_OF_HYDROPHONES == 5:
                elevation_arrays[0,:] = first_elevation

        for i in range(1, NUMBER_OF_FLOATERS):
                if NUMBER_OF_HYDROPHONES == 3:
                        bearing_arrays[i,:] = compute_bearing_angle_array(i + 1)
                elif NUMBER_OF_HYDROPHONES == 4:
                        bearing_arrays[i,:] = compute_bearing_angle_array_square(i + 1)
                else:
                        bearing_arrays[i,:], elevation_arrays[i,:] = compute_bearing_angle_array_complete(i + 1)

                np.save(f"Synth/F{i+1}_azimuth.npy",bearing_arrays[i,:])
                np.save(f"Synth/F{i+1}_elevation.npy",elevation_arrays[i,:])
                
        # Deleting the temporary files and folder
        clean_temporary_files('TMP')

else:
        fist_azimuth = np.load(f"Synth/F1_azimuth.npy")
        first_elevation = np.load(f"Synth/F1_elevation.npy")

        bearing_arrays = np.zeros([NUMBER_OF_FLOATERS,len(fist_azimuth)])
        elevation_arrays = np.zeros([NUMBER_OF_FLOATERS,len(first_elevation)])

        for i in range(NUMBER_OF_FLOATERS):
                bearing_arrays[i,:] = np.load(f"Synth/F{i+1}_azimuth.npy")
                if NUMBER_OF_HYDROPHONES == 5:
                        elevation_arrays[i,:] = np.load(f"Synth/F{i+1}_elevation.npy")



estimated_points = find_points_fixed(RX_Coordinates,bearing_arrays,elevation_arrays)

#estimated_points = replace_outliers_mean(estimated_points,WIN_LEN=7)
#estimated_points = group_close_points(estimated_points,tolleranza_metri=1)
#estimated_points = remove_outliers_median(estimated_points,3)
#estimated_points = np.asarray(clean_trajectory_3d(estimated_points[:,0],estimated_points[:,1],estimated_points[:,2])).T


np.save("Synth/Estimated_Coordinates",estimated_points)


# Plotting points on the map
build_map(
        floaters_coordinates = RX_Coordinates,
        TX_positions_coordinates = TX_Coordinates,
        estimated_vessel_coordinates = estimated_points,
        output_file="map_folium.html",
        track_TX = True,
        track_estimated = True
    )


center = np.load("Synth/Center_Coordinates.npy")

# Finestra di 1000 metri di larghezza e 800 metri di altezza
build_local_cartesian_map(
RX_Coordinates, 
TX_Coordinates, 
estimated_points, 
center_coordinates=center,
window_width_m=600, 
window_height_m=600,
output_file="map_local.png",
track_TX=True,
track_estimated=True
)


if RX_Coordinates.shape[1] == 3 and TX_Coordinates.shape[1] == 3 and estimated_points.shape[1] == 3:
        build_local_cartesian_map_3d(
                RX_Coordinates, 
                TX_Coordinates, 
                estimated_points, 
                center, 
                500, 
                500, 
                max_depth_m=100.0, # Limite dell'asse Z per la visualizzazione
                track_TX=True, 
                track_estimated=True)

print(f"Average estimated TX depth: {np.average(estimated_points[:,2]):1f}")
print(compute_RMSE_same_size(TX_Coordinates[:,:2],estimated_points[:,:2],Lat_center,Lon_center))