import os
import shutil
from dotenv import load_dotenv
from maps.build_local_3D import build_local_cartesian_map_3d
from coordinate_generator import compute_TX_circle_trajectory, geo_to_local, local_to_geo, sposta
from discrete_hydromate_single import run_discrete_hydromate_single
from filter_trajectory import *
from findpoint import *
from maps.build_folium_map import build_map
from maps.build_local_map import build_local_cartesian_map
from ping_all import *
from utils_runner import *
from Floater import *
from PositioningFramework import *

np.random.seed(256123)

SIMULATION_STEPS = 3
NUMBER_OF_FLOATERS = 4

SIMULATE = True
ANALYZE_WAVS = False

RESURFACE_FREQ = 600

load_dotenv()
NUMBER_OF_HYDROPHONES = int(os.getenv('NUMBER_OF_HYDROPHONES'))
SAMPLING_FREQUENCY = int(os.getenv('SAMPLING_FREQUENCY'))

#Center = [20.832813, 88.698390] # India, low depth
Center = [12.61529, 43.37765]
Lat_center = Center[0]
Lon_center = Center[1]



#TX_Coordinates = compute_TX_circle_trajectory(Center, 50 ,start_deg=0,end_deg=350,n_steps=9,radius_m=200,clockwise=True)
#TX_Coordinates = np.zeros((1000,3))
#SIMULATION_STEPS = TX_Coordinates.shape[0]

TX_Coordinates = np.zeros((SIMULATION_STEPS,3))


Center_2 =sposta(Center,150*np.sqrt(2),225)
lat1,lon1 = sposta(Center_2,150,0)
lat2,lon2 = sposta(Center_2,150,90)
lat3,lon3 = sposta(Center_2,150,180)
lat4,lon4 = sposta(Center_2,150,270)
lat5,lon5 = sposta(Center_2,150,225)
RX_init_coordinates = np.asarray( [[lat1,lon1,10],
                                   [lat2,lon2,20],
                                   [lat3,lon3,30],
                                   [lat4,lon4,20],
                                   [lat5,lon5,20]])

# Hydromate is launched from python: this produces three tracks for each floater of "Synth" folder
if SIMULATE:
        RX_gt_Coordinates = np.zeros((SIMULATION_STEPS,NUMBER_OF_FLOATERS,3))
        RX_est_plain_Coordinates = np.zeros((SIMULATION_STEPS,NUMBER_OF_FLOATERS,3))

        if os.path.isdir("Synth"):
                        shutil.rmtree("Synth")
        os.makedirs("Synth")

        np.save("Synth/Center_Coordinates.npy",Center)

        trans = Floater(-100,-100,20,1.0,3)
        trans.set_rho(1.0)
        trans.set_sigma(0.10, 0.10, 0.0)
        trans.set_initial_velocity(1.0,1.0,0)


        # === Floater initialization ===
        floaters = []
        for n in range(NUMBER_OF_FLOATERS):
                # Creation of a new floater
                local_point = geo_to_local(Center,RX_init_coordinates[n])

                f = Floater(
                        gt_x=local_point[0], # Random coordinates
                        gt_y=local_point[1],
                        gt_z=local_point[2],  # Constant depth
                        dt=1.0,
                        dim=3)

                RX_gt_Coordinates[0,n,:] = local_to_geo(Center,f.get_gt_position())
                tmp = np.zeros(RX_gt_Coordinates.shape)

                f.set_initial_velocity(0.3,0.3,0)
                f.set_rho(1.0)
                f.set_sigma(0.1, 0.1, 0.0)
                if n == 4:
                        f.set_initial_velocity(-0.3,0.3,0)
                floaters.append(f)


        PFW = PositioningFramework(RESURFACE_FREQ,np.array([f.get_gt_position() for f in floaters]))
        
        for i in range(SIMULATION_STEPS):
                #print(ping_all(Center,floaters))
                TX_Coordinates[i,:] = local_to_geo(Center,trans.get_gt_position())        
                for n in range(NUMBER_OF_FLOATERS):
                        print(f"# Simulation step {i}, Floater {n} #")
                        RX_gt_Coordinates[i,n,:] = local_to_geo(Center,floaters[n].get_gt_position())
                        tmp[i,n,:] = local_to_geo(Center,floaters[n].get_est_position())

                        if i % RESURFACE_FREQ == 0 and i != 0:
                                print(f"Refurface of floater #{n} at timestamp {i}")
                                floaters[n].resurface()

                        
                        floaters[n].move() # Move the floater of 1 step 
                
                        
                        run_discrete_hydromate_single(TX_Coordinates[i,0],
                                                      TX_Coordinates[i,1],
                                                      TX_Coordinates[i,2],
                                                      RX_gt_Coordinates[i,n,0],
                                                      RX_gt_Coordinates[i,n,1],
                                                      RX_gt_Coordinates[i,n,2],
                                                      (n+1))
                        

                PFW.update_positioning(np.array([f.get_gt_position() for f in floaters]),
                                        np.array([f.get_est_position() for f in floaters]))

                
                                                            
                        
                trans.move()
                print("=#"*10)


        for n in range(NUMBER_OF_FLOATERS):
                for j in range(NUMBER_OF_HYDROPHONES):
                        array = np.load(f'Synth/F{n+1}_H{j+1}.npy')
                        wav.write(f'Synth/F{n+1}_H{j+1}.wav', SAMPLING_FREQUENCY, array)
                                        
                                
                
        RX_fw_IMU_MDS, RX_fw_IMU, RX_bw_IMU_MDS, RX_bw_IMU =  PFW.end_simulation()


        # End of simulation: save coordinates
        np.save("Synth/TX_Coordinates.npy",TX_Coordinates)
        np.save("Synth/RX_gt_Coordinates.npy",RX_gt_Coordinates)

        RX_fw_IMU = local_to_geo(Center,RX_fw_IMU)
        RX_fw_IMU_MDS = local_to_geo(Center,RX_fw_IMU_MDS)
        RX_bw_IMU = local_to_geo(Center,RX_bw_IMU)
        RX_bw_IMU_MDS = local_to_geo(Center,RX_bw_IMU_MDS)

        np.save("Synth/RX_fw_IMU.npy",RX_fw_IMU)
        np.save("Synth/RX_fw_IMU_MDS.npy",RX_fw_IMU_MDS)
        np.save("Synth/RX_bw_IMU.npy",RX_bw_IMU)
        np.save("Synth/RX_bw_IMU_MDS.npy",RX_bw_IMU_MDS)
        
        

#Load coordinates
TX_Coordinates = np.load("Synth/TX_Coordinates.npy")
RX_gt_Coordinates = np.load("Synth/RX_gt_Coordinates.npy")
RX_fw_IMU = np.load("Synth/RX_fw_IMU.npy")
RX_fw_IMU_MDS = np.load("Synth/RX_fw_IMU_MDS.npy")
RX_bw_IMU = np.load("Synth/RX_bw_IMU.npy")
RX_bw_IMU_MDS = np.load("Synth/RX_bw_IMU_MDS.npy")


build_local_cartesian_map(
        RX_gt_Coordinates, 
        None, 
        None, 
        center_coordinates=Center,
        window_width_m=900, 
        window_height_m=900,
        output_file="maps/map_local.png",
        track_TX=True,
        track_estimated=True,
        RX_fw_IMU=RX_fw_IMU,
        RX_fw_IMU_MDS=RX_fw_IMU_MDS,
        RX_bw_IMU=RX_bw_IMU,
        RX_bw_IMU_MDS=RX_bw_IMU_MDS
        )




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
        clean_temporary_files()

else:
        fist_azimuth = np.load(f"Synth/F1_azimuth.npy")
        first_elevation = np.load(f"Synth/F1_elevation.npy")

        bearing_arrays = np.zeros([NUMBER_OF_FLOATERS,len(fist_azimuth)])
        elevation_arrays = np.zeros([NUMBER_OF_FLOATERS,len(first_elevation)])

        for i in range(NUMBER_OF_FLOATERS):
                bearing_arrays[i,:] = np.load(f"Synth/F{i+1}_azimuth.npy")
                if NUMBER_OF_HYDROPHONES == 5:
                        elevation_arrays[i,:] = np.load(f"Synth/F{i+1}_elevation.npy")




estimated_points = find_points(RX_gt_Coordinates,bearing_arrays,elevation_arrays)
estimated_fw_IMU = find_points(RX_fw_IMU,bearing_arrays,elevation_arrays)
estimated_fw_IMU_MDS = find_points(RX_fw_IMU_MDS,bearing_arrays,elevation_arrays)
estimated_bw_IMU = find_points(RX_bw_IMU,bearing_arrays,elevation_arrays)
estimated_bw_IMU_MDS = find_points(RX_bw_IMU_MDS,bearing_arrays,elevation_arrays)

#estimated_points = replace_outliers_mean(estimated_points,WIN_LEN=7)
np.save("Synth/Estimated_Coordinates",estimated_points)
np.save("Synth/Estimated_fw_IMU",estimated_fw_IMU)
np.save("Synth/Estimated_fw_IMU_MDS",estimated_fw_IMU_MDS)
np.save("Synth/Estimated_estimated_bw_IMU",estimated_bw_IMU)
np.save("Synth/Estimated_",estimated_bw_IMU_MDS)

# Plotting points on the map
build_map(
        floaters_coordinates = RX_gt_Coordinates,
        TX_positions_coordinates = TX_Coordinates,
        estimated_vessel_coordinates = estimated_points,
        output_file="maps/map_folium.html",
        track_TX = True,
        track_estimated=True,
        RX_fw_IMU=RX_fw_IMU,
        RX_fw_IMU_MDS=RX_fw_IMU_MDS,
        RX_bw_IMU=RX_bw_IMU,
        RX_bw_IMU_MDS=RX_bw_IMU_MDS
        )


build_local_cartesian_map(
        RX_gt_Coordinates, 
        TX_Coordinates, 
        estimated_fw_IMU, 
        #estimated_points,
        center_coordinates=Center,
        window_width_m=430, 
        window_height_m=430,
        output_file="maps/map_local.png",
        track_TX=True,
        track_estimated=True,
        RX_fw_IMU=RX_fw_IMU,
        #RX_fw_IMU_MDS=RX_fw_IMU_MDS,
        #RX_bw_IMU=RX_bw_IMU,
        #RX_bw_IMU_MDS=RX_bw_IMU_MDS
        )


'''
if RX_gt_Coordinates.shape[2] == 3 and TX_Coordinates.shape[1] == 3 and estimated_points.shape[1] == 3:
        build_local_cartesian_map_3d(
                RX_gt_Coordinates, 
                TX_Coordinates, 
                estimated_points, 
                Center, 
                500, 
                500, 
                max_depth_m=100.0, # Limite dell'asse Z per la visualizzazione
                track_TX=True, 
                track_estimated=True)
'''    

print(f"RMSE estimated_points:\t\t {compute_flat_RMSE(TX_Coordinates[:,:2],estimated_points[:,:2],Lat_center,Lon_center):.1f}")
print(f"RMSE estimated_fw_IMU:\t\t {compute_flat_RMSE(TX_Coordinates[:,:2],estimated_fw_IMU[:,:2],Lat_center,Lon_center):.1f}")
print(f"RMSE estimated_fw_IMU_MDS:\t {compute_flat_RMSE(TX_Coordinates[:,:2],estimated_fw_IMU_MDS[:,:2],Lat_center,Lon_center):.1f}")
print(f"RMSE estimated_bw_IMU: \t\t {compute_flat_RMSE(TX_Coordinates[:,:2],estimated_bw_IMU[:,:2],Lat_center,Lon_center):.1f}")
print(f"RMSE estimated_bw_IMU_MDS: \t {compute_flat_RMSE(TX_Coordinates[:,:2],estimated_bw_IMU_MDS[:,:2],Lat_center,Lon_center):.1f}")
#print(f"RMSE depth: {compute_depth_RMSE(TX_Coordinates[:,2],estimated_points[:,2])}")