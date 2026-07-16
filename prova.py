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
ANALYZE_WAVS = True

dict = {}

load_dotenv()
NUMBER_OF_HYDROPHONES = int(os.getenv('NUMBER_OF_HYDROPHONES'))
SAMPLING_FREQUENCY = int(os.getenv('SAMPLING_FREQUENCY'))

if not os.path.isdir("Synth"):
        os.makedirs("Synth")

for db_d in range (-30,+30):
    db = float(db_d) / 10.0
    # ========== Coordinate generation ==========
    Lat_center, Lon_center = 20.832813, 88.698390
    #Lat_center, Lon_center = 34.7328758990156,-42.736186368656746
    d_RX1 = 10
    d_TX = 50
    #TX_Coordinates = np.asarray([[Lat_center,Lon_center,d_TX]])
    #TX_Coordinates = compute_TX_circle_trajectory(Lat_center, Lon_center,d_TX,0,350,10,200,clockwise=True)


    TX_Coordinates = np.asarray([[Lat_center, Lon_center,d_TX]])
    print(TX_Coordinates)

    SIMULATION_STEPS = TX_Coordinates.shape[0]

 
    lat1,lon1 = sposta(Lat_center,Lon_center,100,0)
    STATIC_RX = np.asarray([[lat1,lon1,d_RX1]])
    #STATIC_RX = np.asarray([[lat1,lon1,d_RX1],[lat2,lon2,d_RX2]])
    RX_Coordinates = np.zeros([SIMULATION_STEPS,STATIC_RX.shape[0],3])
    print(STATIC_RX.shape[0])

    for i in range (SIMULATION_STEPS): # i = timestamp
            for j in range(STATIC_RX.shape[0]): # j = floater
                    RX_Coordinates[i,j,:] = STATIC_RX[j]

    print(RX_Coordinates)
    NUMBER_OF_FLOATERS = RX_Coordinates.shape[1]

    np.save("Synth/TX_Coordinates.npy",TX_Coordinates)
    np.save("Synth/Center_Coordinates.npy",[Lat_center,Lon_center])
    np.save("Synth/RX_Coordinates.npy",RX_Coordinates)

    if TX_Coordinates.shape[1] == 2:
        filler = np.ones(TX_Coordinates.shape[0]).reshape(-1, 1) * 10
        TX_Coordinates = np.hstack((TX_Coordinates, filler))
        print(f"Transmitter array do not have a depth value. Filled with default value of of 10 meters")

    '''
    if RX_Coordinates.shape[1] == 2:
        filler = np.ones(RX_Coordinates.shape[0]).reshape(-1, 1) * 10
        RX_Coordinates = np.hstack((RX_Coordinates, filler))
        print("Floaters array do not have a depth value. Filled with default value of 10 meters")
    '''
            
    # Hydromate is launched from python: this produces three tracks for each floater of "Synth" folder
    if SIMULATE:
            for i in range(NUMBER_OF_FLOATERS):
                    print(f" ## Floater number {i+1} ##")
                    run_discrete_hydromate(TX_Coordinates[:,0],
                                            TX_Coordinates[:,1],
                                            TX_Coordinates[:,2],
                                            RX_Coordinates[:,i,0],
                                            RX_Coordinates[:,i,1],
                                            RX_Coordinates[:,i,2],
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
                    first_bearing,first_elevation = compute_bearing_angle_array_complete(1,DESIRED_SNR=db)
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
                            bearing_arrays[i,:], elevation_arrays[i,:] = compute_bearing_angle_array_complete(i + 1,DESIRED_SNR=db)

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


    center = np.load("Synth/Center_Coordinates.npy")
    dict[db] = bearing_arrays[0]


print()
print()
for k,v in dict.items():
        print(f"{k}db - {v[0]}")