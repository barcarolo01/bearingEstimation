import os
import shutil
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from coordinate_generator import *
from discrete_hydromate_single import run_discrete_hydromate_single
from filter_trajectory import *
from findpoint import *
from maps.build_folium_map import build_map
from maps.build_local_map import build_local_cartesian_map
from ping_all import *
from utils_runner import *
from Floater import *

np.random.seed(256123)

NUMBER_OF_FLOATERS = 2


load_dotenv()
NUMBER_OF_HYDROPHONES = int(os.getenv('NUMBER_OF_HYDROPHONES'))
SAMPLING_FREQUENCY = int(os.getenv('SAMPLING_FREQUENCY'))

#Center = [20.832813, 88.698390] # India, low depth
#Center = [12.61529, 43.37765]
Center = [39.84164446886851,-70.90061264492535]

TX_Coordinates = generate_grid_of_samples(Center[0],Center[1],10,500,20)
TX_constant_depth = 10

SIMULATION_STEPS = TX_Coordinates.shape[0]
TX_Coordinates = np.asarray([TX_Coordinates[:,0],TX_Coordinates[:,1],np.ones(SIMULATION_STEPS)*TX_constant_depth]).T

lat1,lon1 = sposta(Center,150,270)
lat2,lon2 = sposta(Center,150,90)
RX_init_coordinates = np.asarray( [[lat1,lon1,10],
                                   [lat2,lon2,10]])

np.save("RX_static_coordinates.npy",RX_init_coordinates)
fw = open("accuracy.csv","w")
fw.write("GT_Lat,GT_Lon,GT_Depth,Est_Lat,Est_Lon,Est_depth\n")

for i in range(SIMULATION_STEPS):
        # Skip TX positions too close to a receiver to avoid Bellhop errors
        for n in range(NUMBER_OF_FLOATERS):
                if calculate_distance(TX_Coordinates[i,:2],RX_init_coordinates[n,:2]) < 1:
                        continue

        if os.path.isdir("Synth"):
                shutil.rmtree("Synth")
        os.makedirs("Synth")
        print(f"Simulation step n. {i+1}/{SIMULATION_STEPS}")

        for n in range(NUMBER_OF_FLOATERS):
                run_discrete_hydromate_single(TX_Coordinates[i,0],
                                        TX_Coordinates[i,1],
                                        TX_Coordinates[i,2],
                                        RX_init_coordinates[n,0],
                                        RX_init_coordinates[n,1],
                                        RX_init_coordinates[n,2],
                                        (n+1))
                                
                for j in range(NUMBER_OF_HYDROPHONES):
                        array = np.load(f'Synth/F{n+1}_H{j+1}.npy')
                        wav.write(f'Synth/F{n+1}_H{j+1}.wav', SAMPLING_FREQUENCY, array)
                                                
                                
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
        np.save(f"Synth/F1_azimuth.npy",first_bearing)
        
        if NUMBER_OF_HYDROPHONES == 5:
                elevation_arrays[0,:] = first_elevation

        for j in range(1, NUMBER_OF_FLOATERS):
                if NUMBER_OF_HYDROPHONES == 3:
                        bearing_arrays[j,:] = compute_bearing_angle_array(j + 1)
                elif NUMBER_OF_HYDROPHONES == 4:
                        bearing_arrays[j,:] = compute_bearing_angle_array_square(j + 1)
                else:
                        bearing_arrays[j,:], elevation_arrays[j,:] = compute_bearing_angle_array_complete(j + 1)

                np.save(f"Synth/F{j+1}_azimuth.npy",bearing_arrays[j,:])
                np.save(f"Synth/F{j+1}_elevation.npy",elevation_arrays[j,:])
                
        # Deleting the temporary files and folder
        clean_temporary_files()
        
        estimated_points = find_points(np.asarray([RX_init_coordinates]),bearing_arrays,elevation_arrays)
        fw.write(f"{TX_Coordinates[i,0]},{TX_Coordinates[i,1]},{TX_Coordinates[i,2]},{estimated_points[0,0]},{estimated_points[0,1]},{estimated_points[0,2]}\n")
        shutil.rmtree("Synth")