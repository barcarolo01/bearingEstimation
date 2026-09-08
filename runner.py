import os
import shutil
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from coordinate_generator import geo_to_local, local_to_geo, sposta
from discrete_hydromate_single import run_discrete_hydromate_single
from filter_trajectory import *
from findpoint import *
from maps.build_folium_map import build_map
from maps.build_local_map import build_local_cartesian_map
from ping_all import *
from utils_runner import *
from Floater import *

np.random.seed(256123)

SIMULATION_STEPS = 1 
RESURFACE_FREQ = 999999
NUMBER_OF_FLOATERS = 2
TX_LIMIT = 2

SIMULATE = True
SKIPhydromate = False
ANALYZE_WAVS = False

load_dotenv()
NUMBER_OF_HYDROPHONES = int(os.getenv('NUMBER_OF_HYDROPHONES'))
SAMPLING_FREQUENCY = int(os.getenv('SAMPLING_FREQUENCY'))

#Center = [20.832813, 88.698390] # India, low depth
#Center = [12.61529, 43.37765]
Center = [39.84164446886851,-70.90061264492535]


Lat_center = Center[0]
Lon_center = Center[1]

#TX_Coordinates = compute_TX_circle_trajectory(Center, 50 ,start_deg=0,end_deg=350,n_steps=9,radius_m=200,clockwise=True)
#TX_Coordinates = np.zeros((1000,3))
#SIMULATION_STEPS = TX_Coordinates.shape[0]

TX_Coordinates = np.zeros((SIMULATION_STEPS,3))

Center_2 = sposta(Center,150*np.sqrt(2),225)
lat1,lon1 = sposta(Center,100,0)
lat2,lon2 = sposta(Center,100,90)
lat3,lon3 = sposta(Center_2,201,180)
lat4,lon4 = sposta(Center_2,250,270)
lat5,lon5 = sposta(Center_2,50,200)

RX_init_coordinates = np.asarray( [[lat1,lon1,10],
                                   [lat2,lon2,20],
                                   [lat3,lon3,30],
                                   [lat4,lon4,20],
                                   [lat5,lon5,10]])

# Hydromate is launched from python: this produces three tracks for each floater of "Synth" folder
if SIMULATE:
        RX_gt_Coordinates = np.zeros((SIMULATION_STEPS,NUMBER_OF_FLOATERS,3))
        RX_est_plain_Coordinates = np.zeros((SIMULATION_STEPS,NUMBER_OF_FLOATERS,3))

        if os.path.isdir("Synth"):
                shutil.rmtree("Synth")
        os.makedirs("Synth")

        np.save("Synth/Center_Coordinates.npy",Center)

        trans = Floater(0,0,0,20,NUMBER_OF_FLOATERS,1.0,3)
        trans.set_rho(1.0)
        trans.set_sigma(0.10, 0.10, 0.0)
        trans.set_initial_velocity(1.0,1.0,0)

        # === Floater initialization ===
        floaters = []
        for n in range(NUMBER_OF_FLOATERS):
                # Creation of a new floater
                local_point = geo_to_local(Center,RX_init_coordinates[n])

                f = Floater(
                        ID = n,
                        gt_x=local_point[0], # Random coordinates
                        gt_y=local_point[1],
                        gt_z=local_point[2],  # Constant depth
                        NF=NUMBER_OF_FLOATERS,
                        dt=1.0,
                        dim=3)

                RX_gt_Coordinates[0,n,:] = local_to_geo(Center,f.gt_pos)
                tmp = np.zeros(RX_gt_Coordinates.shape)

                f.set_initial_velocity(0.3,0.3,0)
                f.set_rho(1)
                f.set_sigma(0.1, 0.1, 0.0)
                if n == 4:
                        f.set_initial_velocity(-0.3,0.3,0)
                floaters.append(f)

        round_active  = False
        round_order   = []     # sequenza di trasmettitori, lunga 2N-1
        round_step    = 0
        round_id      = 0
        already_fired = set()  # ID dei nodi che hanno già innescato un round

        resurf_dict = {}
        for i in range(SIMULATION_STEPS):
                transmissions_in_round = 0
                TX_Coordinates[i,:] = local_to_geo(Center,trans.gt_pos)        


                mustTX = [False] * NUMBER_OF_FLOATERS
                for n in range(NUMBER_OF_FLOATERS):
                        #print(f"# Simulation step {i}, Floater {n} #")
                        RX_gt_Coordinates[i,n,:] = local_to_geo(Center,floaters[n].gt_pos)
                        tmp[i,n,:] = local_to_geo(Center,floaters[n].est_pos)

                        if (i % RESURFACE_FREQ == 0 and i != 0) or  (i == SIMULATION_STEPS -1):
                                floaters[n].resurface()

                        mustTX[n] = floaters[n].move()

                        # At the end of the simulation, the floater emerges
                        if( i == SIMULATION_STEPS - 1):
                                floaters[n].resurface()


                        
                        # Hydromate simulation
                        if not SKIPhydromate:
                                run_discrete_hydromate_single(TX_Coordinates[i,0],
                                                        TX_Coordinates[i,1],
                                                        TX_Coordinates[i,2],
                                                        RX_gt_Coordinates[i,n,0],
                                                        RX_gt_Coordinates[i,n,1],
                                                        RX_gt_Coordinates[i,n,2],
                                                        (n+1))
                        
                trans.move() # Vessel motion
           

                # Current shapshot
                positions = np.array([f.gt_pos for f in floaters])
                clocks    = np.array([f.clk    for f in floaters])

                # If the network is not already performing a ranging operation
                if not round_active:
                        triggers = []

                        for n in range(NUMBER_OF_FLOATERS):
                                if mustTX[n]:# and floaters[n].ID not in already_fired:
                                        triggers.append(n)


                        if triggers:
                                print(f"[step {i}] TRIGGER → round {round_id+1}")
                                t = triggers[0] # The first node triggering the ranging become the round initiator
                                #for k in triggers:
                                        #already_fired.add(floaters[k].ID)     

                                round_id  += 1 # Increment MDS round counter

                                # The triggering node is the first in the schedule, then all others transmit according to ID order
                                order = [t]
                                for k in range(NUMBER_OF_FLOATERS):
                                        if k != t:
                                                order.append(k)
                        
                                # Complete round order (2N-1 transmissions, the initiator transmits only once)
                                round_order = order + order[:-1]

                                # In-round counters
                                round_step  = 0
                                round_active = True

                                # Update the rangind round ID for each floater
                                for f in floaters:          
                                        f.start_round(round_id)

                                print(f"[Sim step {i}]: Rangin round {round_id} triggered by floater {floaters[t].ID}")

   
                # If a ranging round is ongoing
                if round_active:
                        for k in range(TX_LIMIT):
                                if round_step >= len(round_order):
                                        break
                                tx   = round_order[round_step] # ID of the scheduled transmitter
                                t_tx = clocks[tx] + math.ceil(1000/TX_LIMIT)
                                print(f"SIMSTEP {i}, transmission of {tx}")
                                payload = floaters[tx].on_transmit(t_tx, tx)

                                floaters[tx].events.append({
                                        "t_local": t_tx, "type": "TX",
                                        "src": floaters[tx].ID, "payload": payload,
                                })


                                # Register the reception on the local event log of all other floaters
                                for m in range(NUMBER_OF_FLOATERS):
                                        if m == tx:
                                                continue

                                        # TODO: Compute the distance using hydromate
                                        d    = ((np.linalg.norm(positions[tx] - positions[m]))/1500) * 1000

                                        t_rx = clocks[m] + d + math.ceil(1000/TX_LIMIT)
                                        floaters[m].on_receive(t_rx, tx, m, payload, floaters[tx].ID)
                                        

                                round_step += 1
                        if round_step == len(round_order):
                                round_active = False
                                print(f"SIMSTEP {i}, ended rangeing")
                                



                
                

        RX_fw_IMU     = np.zeros((SIMULATION_STEPS, NUMBER_OF_FLOATERS, 3))
        RX_fw_IMU_MDS = np.zeros((SIMULATION_STEPS, NUMBER_OF_FLOATERS, 3))
        RX_bw_IMU     = np.zeros((SIMULATION_STEPS, NUMBER_OF_FLOATERS, 3))
        RX_bw_IMU_MDS = np.zeros((SIMULATION_STEPS, NUMBER_OF_FLOATERS, 3))
        RX_IMU_compensated = np.zeros((SIMULATION_STEPS, NUMBER_OF_FLOATERS, 3))
        GT            = np.zeros((SIMULATION_STEPS, NUMBER_OF_FLOATERS, 3))

        names  = ['imu', 'imu_rev', 'imu_mds', 'imu_mds_rev', 'imu_compensated','imu_compensated_bis']
        colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', '#000000']
        targets = [RX_fw_IMU, RX_bw_IMU, RX_fw_IMU_MDS, RX_bw_IMU_MDS,RX_IMU_compensated]
        for n in range(NUMBER_OF_FLOATERS):
                res = floaters[n].return_results(gps_sigma=0, fuse=False)
                gt  = res['gt'][1:]
                GT[:, n, :] = gt

                fig, (ax_pred, ax_true) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

                for name, color, arr in zip(names, colors, targets):
                        pos, err = res[name]
                        pos, err = pos[1:], err[1:]

                        arr[:, n, :] = pos

                        ax_pred.plot(err, color=color, lw=1.8, label=name)
                        ax_true.plot(np.linalg.norm(pos - gt, axis=1), color=color, lw=1.8, label=name)
                        

                for ax, t in ((ax_pred, "Errore auto-stimato"), (ax_true, "Errore vero vs ground truth")):
                        ax.set_title(t)
                        ax.set_xlabel("Passi di simulazione")
                        #ax.set_ylim([0,500])
                        ax.grid(True, ls='--', alpha=.5)
                        for k in floaters[n].Resurface_index:
                                if 1 <= k <= SIMULATION_STEPS:
                                        ax.axvline(k - 1, color='gray', ls=':', lw=.8)

                ax_pred.set_ylabel("Metri")
                ax_true.legend(loc="upper right", frameon=True)

                fig.suptitle(f"Floater {n}", fontsize=14)
                fig.tight_layout()
                plt.savefig(f"floater_{n}_errors.png", dpi=120)
                plt.close(fig)

        RX_fw_IMU = local_to_geo(Center,RX_fw_IMU)
        RX_fw_IMU_MDS = local_to_geo(Center,RX_fw_IMU_MDS)
        RX_bw_IMU = local_to_geo(Center,RX_bw_IMU)
        RX_bw_IMU_MDS = local_to_geo(Center,RX_bw_IMU_MDS)
        RX_IMU_compensated = local_to_geo(Center,RX_IMU_compensated)

        '''
        for l in range(380,420):
                print(f"tmpstmp{l}: {np.linalg.norm(Casss[l,0,:]-GT[l,0,:])}")
        '''



        
        print(f"{'Version':22s} {'RMSE':>10s}")
        for name, arr in zip(names, targets):
                e_all, p_all = [], []
                for n in range(NUMBER_OF_FLOATERS):
                        res = floaters[n].return_results(gps_sigma=0,fuse=False)
                        pos, err = res[name][0][1:], res[name][1][1:]
                        e_all.append(np.linalg.norm(pos - res['gt'][1:], axis=1))
                        p_all.append(err)
                e = np.concatenate(e_all); p = np.concatenate(p_all) 
                print(f"{name:22s} {np.sqrt((e**2).mean()):10.3f}")
        

        '''
        build_local_cartesian_map(
                RX_gt_Coordinates,
                None, 
                None, 
                center_coordinates=Center,
                window_width_m=430, 
                window_height_m=430,
                output_file="maps/map_local.png",
                track_TX=True,
                track_estimated=True,
                RX_fw_IMU=RX_fw_IMU,
                RX_fw_IMU_MDS=RX_fw_IMU_MDS,
                RX_bw_IMU=RX_bw_IMU,
                RX_bw_IMU_MDS=RX_bw_IMU_MDS
                )

        
        build_map(
                floaters_coordinates = RX_gt_Coordinates,
                TX_positions_coordinates = None,
                estimated_vessel_coordinates = None,
                output_file="maps/map_folium.html",
                track_TX = True,
                track_estimated=True,
                RX_fw_IMU=RX_fw_IMU,
                RX_fw_IMU_MDS=RX_IMU_compensated,
                RX_bw_IMU=RX_bw_IMU,
                #RX_bw_IMU_MDS=RX_bw_IMU_MDS
                )
        '''


        for n in range(NUMBER_OF_FLOATERS):
                for j in range(NUMBER_OF_HYDROPHONES):
                        array = np.load(f'Synth/F{n+1}_H{j+1}.npy')
                        wav.write(f'Synth/F{n+1}_H{j+1}.wav', SAMPLING_FREQUENCY, array)
                                        
                                


        # End of simulation: save coordinates
        np.save("Synth/TX_Coordinates.npy",TX_Coordinates)
        np.save("Synth/RX_gt_Coordinates.npy",RX_gt_Coordinates)

        

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
        RX_fw_IMU_MDS=RX_IMU_compensated,
        RX_bw_IMU=RX_bw_IMU,
        #RX_bw_IMU_MDS=RX_bw_IMU_MDS
        )

'''
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
#print(f"RMSE estimated_fw_IMU:\t\t {compute_flat_RMSE(TX_Coordinates[:,:2],estimated_fw_IMU[:,:2],Lat_center,Lon_center):.1f}")
#print(f"RMSE estimated_fw_IMU_MDS:\t {compute_flat_RMSE(TX_Coordinates[:,:2],estimated_fw_IMU_MDS[:,:2],Lat_center,Lon_center):.1f}")
#print(f"RMSE estimated_bw_IMU: \t\t {compute_flat_RMSE(TX_Coordinates[:,:2],estimated_bw_IMU[:,:2],Lat_center,Lon_center):.1f}")
#print(f"RMSE estimated_bw_IMU_MDS: \t {compute_flat_RMSE(TX_Coordinates[:,:2],estimated_bw_IMU_MDS[:,:2],Lat_center,Lon_center):.1f}")
#print(f"RMSE depth: {compute_depth_RMSE(TX_Coordinates[:,2],estimated_points[:,2])}")
