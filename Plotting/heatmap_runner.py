import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from dotenv import load_dotenv
from coordinate_generator import  generate_grid_of_samples,calculate_distance, sposta
from discrete_hydromate import run_discrete_hydromate
from findpoint import *
from build_folium_map import build_map
from utils_runner import *

SIMULATE = True
W = 300
N = 15


Lat_center, Lon_center = 20.832813, 88.698390

Coordinate_SAMPLED = generate_grid_of_samples(lat_orig=Lat_center,lon_orig=Lon_center,W=W,N=N)
np.save("Synth/TX_Coordinates.npy",Coordinate_SAMPLED)
lat1,lon1 = sposta(Lat_center,Lon_center,100,90)
lat2,lon2 = sposta(Lat_center,Lon_center,100,270)
RX_Coordinates = np.asarray([[lat1,lon1],[lat2,lon2]])
np.save("Synth/RX_Coordinates.npy",RX_Coordinates)

load_dotenv()
NUMBER_OF_FLOATERS = 2
NUMBER_OF_HYDROPHONES = 3
SAMPLING_FREQUENCY = int(os.getenv('SAMPLING_FREQUENCY'))


if not os.path.isdir("Synth"):
        os.makedirs("Synth")

f = open("Plotting/grid_errors_for_heatmap.csv", "w")
f.write("Lat_GT,Lon_GT,Avg_Lat_Est,Avg_Lon_Est,Avg_distance,RMSE\n")

for TX_sample_point in Coordinate_SAMPLED:
        dist_F1_TX = calculate_distance(TX_sample_point,RX_Coordinates[0,:2])
        dist_F2_TX = calculate_distance(TX_sample_point,RX_Coordinates[1,:2])
        # Skip points too close to one of the floaters
        if dist_F1_TX < 2 or dist_F2_TX < 2:
                print("SKIPPED")
                continue

        TX_Coordinates = np.asarray([TX_sample_point])

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

        #intersection_points = find_points(RX_Coordinates,bearing_arrays,elevation_arrays)
        intersection_points = find_points(RX_Coordinates,bearing_arrays,elevation_arrays)
        #intersection_points = filter_trajectory_outliers(intersection_points, window_size=8, max_distance=5)

        # Plotting points on the map
        build_map(
        floaters_coordinates = RX_Coordinates,
        TX_positions_coordinates = TX_Coordinates,
        estimated_vessel_coordinates = intersection_points,
        output_file="map.html",
        track_TX = True,
        track_estimated = False
        )
        avg_error = 0
        squared_sum = 0
        number = 0
        for i in range(int(intersection_points.shape[0])):
                dist = calculate_distance(TX_Coordinates[0,:2],intersection_points[i,:2])
                if dist is not np.nan:
                        avg_error += dist       
                        squared_sum += (dist*dist)
                        number += 1

        avg_error = avg_error / number
        RMSE = np.sqrt(squared_sum/number)

        avg_coordinates = np.mean(intersection_points,axis=0)
        coordinates = np.asarray([[avg_coordinates[0],avg_coordinates[1]]])
        avg_distance = calculate_distance(TX_Coordinates[0,:2],coordinates[0])
        f.write(f"{TX_Coordinates[0,0]},{TX_Coordinates[0,1]},{avg_coordinates[0]},{avg_coordinates[1]},{avg_error},{RMSE}\n")
        f.flush()

f.close()