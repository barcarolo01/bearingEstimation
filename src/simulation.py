import os
import shutil
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from digitalshadow.Positioning.coordinate_generator import *
from digitalshadow.underwater_sim.discrete_hydromate_single import *
from digitalshadow.Positioning.point_estimation import *
from digitalshadow.maps.build_local_map import build_local_cartesian_map
from digitalshadow.maps.map_common import Track
from digitalshadow.underwater_sim.floater_ping import *
from digitalshadow.utils.utils import *
from digitalshadow.devices.Floater import *
from digitalshadow.Positioning.kalman_filter import *
from digitalshadow.Positioning.tracking_plots import *

FONTSIZE = 18
PACKET_LOSS = 0 # Packet loss factor (between 0 and 1)

ADD_PSI_ERROR = True
HYDROMATE_SIMULATION = False

colors = {
                'imu':  "#0000FF",
                'imu_mds':  "#2AB040",
                'imu_rev':  "#FC03D3",
                #'imu_mds_rev':  "#D6940F",
                'imu_compensated': "#D6940F"
         }


class Simulation:
        def __init__(self, Floaters, Center, transmitter=None,  Steps=1, transmitter_coordinates=None, sim_name="simulation0", TX_LIMIT=1, seed=256123):
                '''
                Initialization method. Input:
                 * Floaters: list of 'Floater' objects
                 * Steps: number of simulation steps
                 * Center: 2D Coordinates of the center (numpy array of shape (2,))
                 * transmitter: 'Transmitter' object 
                 * Steps: number of simulation steps (used only if 'transmitter' is passed)
                 * transmitter_coordinates: numpy array of 3D local coordinates shape (I,3), 
                        I = number of steps
                        3 = (latitude,longitude,depth)
                 * seed: simulation seed

                Only one between 'transmitter' and 'transmitter_coordinates' must be passed to the constructor.
                '''   
      
                load_dotenv()
                self.rnd_packetloss = np.random.default_rng(seed+1)

                self.sim_name = sim_name
                if HYDROMATE_SIMULATION or True:
                        if os.path.isdir(self.sim_name):
                                shutil.rmtree(self.sim_name)
                        os.makedirs(self.sim_name)

                '''
                if os.path.isdir(self.sim_name):
                        raise NameError(f"A folder named {self.sim_name} already exist. Change simulation name or delete the folder.")
                else:
                        os.makedirs(self.sim_name)
                '''
                
                self.NUMBER_OF_FLOATERS = len(Floaters)
                self.Center = Center
                self.Floaters = Floaters
                self.TX_LIMIT = TX_LIMIT
                for f in Floaters:
                        f.TX_LIMIT = TX_LIMIT
                self.NUMBER_OF_HYDROPHONES = int(os.getenv('NUMBER_OF_HYDROPHONES'))
                self.SAMPLING_FREQUENCY = int(os.getenv('SAMPLING_FREQUENCY'))


                if transmitter is None and transmitter_coordinates is None:
                        raise ValueError(f"Both transmitter and transmitter_coordinates are None")
                if transmitter is not None and transmitter_coordinates is not None:
                        raise ValueError(f"Both transmitter and transmitter_coordinates are provided. Use only one of them.")
                if transmitter is not None:
                        self.transmitter = transmitter
                        self.SIMULATION_STEPS = Steps
                        self.TX_positions = np.zeros((self.SIMULATION_STEPS,3))
                if transmitter_coordinates is not None:
                        self.transmitter = None
                        self.SIMULATION_STEPS = transmitter_coordinates.shape[0]
                        self.TX_positions = transmitter_coordinates.copy()
                        
                self.RX_positions = np.zeros((self.SIMULATION_STEPS,self.NUMBER_OF_FLOATERS,3))
                        
                # Floater data for vessel DoA estimation
                self.bearing_arrays = np.full((self.NUMBER_OF_FLOATERS,self.SIMULATION_STEPS), np.nan)
                self.elevation_arrays = np.full((self.NUMBER_OF_FLOATERS,self.SIMULATION_STEPS), np.nan)
                self.psi_error_arrays = np.full((self.NUMBER_OF_FLOATERS,self.SIMULATION_STEPS), np.nan)

        
        def run_simulation(self):
                RX_fw_IMU     = np.zeros((self.SIMULATION_STEPS, self.NUMBER_OF_FLOATERS, 3))
                RX_fw_IMU_MDS = np.zeros((self.SIMULATION_STEPS, self.NUMBER_OF_FLOATERS, 3))
                RX_bw_IMU     = np.zeros((self.SIMULATION_STEPS, self.NUMBER_OF_FLOATERS, 3))
                RX_bw_IMU_MDS = np.zeros((self.SIMULATION_STEPS, self.NUMBER_OF_FLOATERS, 3))
                RX_IMU_compensated = np.zeros((self.SIMULATION_STEPS, self.NUMBER_OF_FLOATERS, 3))

                targets = {
                        'imu':  RX_fw_IMU,
                        'imu_mds':  RX_fw_IMU_MDS,
                        'imu_rev':  RX_bw_IMU,
                        'imu_mds_rev':  RX_bw_IMU_MDS,
                        'imu_compensated': RX_IMU_compensated
                }
                GT = np.zeros((self.SIMULATION_STEPS, self.NUMBER_OF_FLOATERS, 3))
                RX_positions = np.zeros((self.SIMULATION_STEPS,self.NUMBER_OF_FLOATERS,3))

                if os.path.isdir("Synth"):
                        shutil.rmtree("Synth")
                os.makedirs("Synth")
                

                round_active  = False
                round_order   = []     # Floater transmitting sequence (2N-1 long)
                round_step    = 0
                round_id      = 0

                mustTX = [False] * self.NUMBER_OF_FLOATERS
                for i in range(self.SIMULATION_STEPS):
                        #print(f"# Simulation step {i+1}/{self.SIMULATION_STEPS}")
                        if self.transmitter is None:
                                TX_Coordinates = local_to_geo(self.Center,self.TX_positions[i,:])
                        else:
                                TX_Coordinates = local_to_geo(self.Center,self.transmitter.gt_pos)

                        for n in range(self.NUMBER_OF_FLOATERS):
                                self.RX_positions[i,n,:] = self.Floaters[n].gt_pos
                                RX_gt_Coordinates = local_to_geo(self.Center,self.RX_positions[i,n,:])

                                # Hydromate simulation
                                if HYDROMATE_SIMULATION:
                                        run_discrete_hydromate_single(  TX_Coordinates[0],
                                                                        TX_Coordinates[1],
                                                                        TX_Coordinates[2],
                                                                        RX_gt_Coordinates[0],
                                                                        RX_gt_Coordinates[1],
                                                                        RX_gt_Coordinates[2],
                                                                        (n+1),
                                                                        PSI_RX=self.Floaters[n].gt_psi )

                                        # Save as wav segment
                                        for j in range(self.NUMBER_OF_HYDROPHONES):
                                                hydrophone_track = np.load(os.path.join('TMP',f'H{j+1}.npy'))
                                                wav.write(f'Synth/T{i}_F{n+1}_H{j+1}.wav', self.SAMPLING_FREQUENCY, hydrophone_track)


                                        if NUMBER_OF_HYDROPHONES == 3:
                                                self.bearing_arrays[n,i], self.elevation_arrays[n,i]  = compute_single_bearing_angle_triangle(timestamp=i, wav_folder='Synth',F_index=(n + 1))
                                        if NUMBER_OF_HYDROPHONES == 4:
                                                self.bearing_arrays[n,i], self.elevation_arrays[n,i]  = compute_single_bearing_angle_square(timestamp=i, wav_folder='Synth',F_index=(n + 1))
                                        if NUMBER_OF_HYDROPHONES == 5:
                                                self.bearing_arrays[n,i], self.elevation_arrays[n,i]  = compute_single_bearing_angle_complete(timestamp=i, wav_folder='Synth',F_index=(n + 1))
                                        
                        # Current clock shapshot
                        clocks    = np.array([f.clk    for f in self.Floaters])

                        # If the network is not already performing a ranging operation
                        if not round_active:
                                triggers = []

                                for n in range(self.NUMBER_OF_FLOATERS):
                                        if mustTX[n]:
                                                triggers.append(n)

                                if triggers:
                                        print(f"[step {i}] TRIGGER → round {round_id+1}")
                                        t = triggers[0] # The first node triggering the ranging become the round initiator

                                        round_id  += 1 # Increment MDS round counter

                                        # The triggering node is the first in the schedule, then all others transmit according to ID order
                                        order = [t]
                                        for k in range(self.NUMBER_OF_FLOATERS):
                                                if k != t:
                                                        order.append(k)
                                
                                        # Complete round order (2N-1 transmissions, the initiator transmits only once)
                                        round_order = order + order[:-1]

                                        # In-round counters
                                        round_step  = 0
                                        round_active = True

                                        # Update the rangind round ID for each floater
                                        for f in self.Floaters:          
                                                f.start_round(round_id)

                                        print(f"[Sim step {i}]: Rangin round {round_id} triggered by floater {self.Floaters[t].ID}")

        
                        # If a ranging round is ongoing
                        if round_active:
                                for k in range(self.TX_LIMIT):
                                        if round_step >= len(round_order):
                                                break
                                        tx   = round_order[round_step] # ID of the scheduled transmitter
                                        t_tx = clocks[tx] + math.ceil(1000/self.TX_LIMIT)
                                        print(f"SIMSTEP {i}, transmission of {tx}")
                                        payload = self.Floaters[tx].on_transmit(t_tx, tx)

                                        self.Floaters[tx].events.append({
                                                "t_local": t_tx, "type": "TX",
                                                "src": self.Floaters[tx].ID, "payload": payload,
                                        })


                                        # Register the reception on the local event log of all other floaters
                                        tot = 0
                                        lost = 0
                                        for m in range(self.NUMBER_OF_FLOATERS):
                                                if m == tx:
                                                        continue

                                                if self.rnd_packetloss.uniform(0,1) >= PACKET_LOSS:
                                                        #delay_ms = ping_pair(local_to_geo(self.Center,self.Floaters[tx].gt_pos),local_to_geo(self.Center,self.Floaters[m].gt_pos))
                                                        delay_ms = 1000* np.linalg.norm(self.Floaters[tx].gt_pos[:2]-self.Floaters[m].gt_pos[:2]) / 1500
                                                        t_rx = clocks[m] + delay_ms + math.ceil(1000/self.TX_LIMIT)
                                                        self.Floaters[m].on_receive(t_rx, tx, m, payload, self.Floaters[tx].ID)
                                                else:
                                                        #print("LOSS")
                                                        lost+=1
                                                tot+=1
                                        print(f"LOST {lost}/{tot}")
                                                
                                        round_step += 1
                                if round_step == len(round_order):
                                        round_active = False
                                        print(f"SIMSTEP {i}, ended rangeing")

                        # Advance the simulation for the next step
                        mustTX = [False] * self.NUMBER_OF_FLOATERS
                        for n in range(self.NUMBER_OF_FLOATERS):
                                # At the end of the simulation, the floater emerges
                                if i == (self.SIMULATION_STEPS-1):
                                        self.Floaters[n].resurface()
                                else:
                                        mustTX[n] = self.Floaters[n].move()
                                        
                        if self.transmitter is not None: 
                                self.transmitter.move() # Vessel motion


                # Save simulation results
                np.save(os.path.join(self.sim_name,"RXs.npy"),self.RX_positions)
                np.save(os.path.join(self.sim_name,"TXs.npy"),self.TX_positions)
                np.save(os.path.join(self.sim_name,"bearings.npy"),self.bearing_arrays)

                names   = ['imu', 'imu_mds', 'imu_compensated']
                results = [] 

                self.RX_positions = np.load(os.path.join(self.sim_name,"RXs.npy"))
                self.TX_positions = np.load(os.path.join(self.sim_name,"TXs.npy"))
                self.bearing_arrays = np.load(os.path.join(self.sim_name,"bearings.npy"))
                for n in range(self.NUMBER_OF_FLOATERS):
                        res = self.Floaters[n].return_results(fuse=False)

                        self.psi_error_arrays[n,:] = res['psi_err'].copy()
                        results.append(res)

                        gt = res['gt']
                        GT[:, n, :] = gt

                        fig, ax = plt.subplots(figsize=(9, 5))

                        for j,name in enumerate(names):
                                pos = res[name]
                                targets[name][:, n, :] = pos
                                ax.plot(np.linalg.norm(pos - gt, axis=1), color=colors[name], lw=1.8, label=name)

                        ax.set_title("Positioning error vs ground truth")
                        ax.set_xlabel("Simulation steps")
                        ax.set_ylabel("Meters")
                        ax.grid(True, ls='--', alpha=.5)
                        for k in self.Floaters[n].Resurface_index:
                                if 1 <= k <= self.SIMULATION_STEPS:
                                        ax.axvline(k - 1, color='gray', ls=':', lw=.8)
                        ax.legend(loc="upper right", frameon=True)
                        
                        fig.suptitle(f"Floater {n}", fontsize=14)
                        fig.tight_layout()
                        plt.savefig(os.path.join(self.sim_name,f"floater_{n}_errors.png"), dpi=300)
                        plt.close(fig)
                
                print(f"{'Version':22s} {'RMSE':>10s}")
                for name in names:
                        e_all = []
                        for n in range(self.NUMBER_OF_FLOATERS):
                                res = results[n]
                                pos = res[name]
                                e_all.append(np.linalg.norm(pos - res['gt'], axis=1))
                        e = np.concatenate(e_all)
                        print(f"{name:22s} {np.sqrt((e**2).mean()):10.3f}")

                clean_temporary_files()

                if ADD_PSI_ERROR and HYDROMATE_SIMULATION:
                        print(f"psi_error_arrays: {self.psi_error_arrays}")
                        self.bearing_arrays = wrap_degrees(self.bearing_arrays + self.psi_error_arrays)
                        print(f"FINAL BEARINGS: {self.bearing_arrays}")

                # TODO modifica
                #res = kalman_track(self.RX_positions, self.bearing_arrays, dt=4.5,elevation_array=self.elevation_arrays)
                #tracking_report(res, self.TX_positions)
                #plot_tracking(res, self.TX_positions, self.RX_positions,save_path="trackNEW.png")

                # TODO: TMP
                #estimated_points = find_points(self.RX_positions,self.bearing_arrays,self.elevation_arrays)

                # Plotting points on the map 
                '''
                build_local_cartesian_map(
                        floater_coordinates=self.RX_positions,
                        TX_coordinates=self.TX_positions,
                        estimated_vessel_coordinates=res['raw'],
                        tracks=[
                                Track("RX IMU", RX_fw_IMU, "#0000FF"),
                                Track("RX IMU+MDS", RX_fw_IMU_MDS, "#2AB040"),
                                Track("Compensated", RX_IMU_compensated, "#FF8822"),
                                ],
                        output_file=os.path.join(self.sim_name,"local_map.png"),
                        LEGEND=False
                )

                build_local_cartesian_map(
                                        floater_coordinates=self.RX_positions,
                                        TX_coordinates=self.TX_positions,
                                        estimated_vessel_coordinates=res['smoothed'],
                                        output_file=os.path.join(self.sim_name,"local_map_smoothed.png"),
                                        LEGEND=False
                                )
                '''

                return {
                        "RX_gt_pos": self.RX_positions,
                        "TX_gt_pos": self.TX_positions,
                        "RX_imu": RX_fw_IMU,
                        "RX_imu_mds": RX_fw_IMU_MDS,
                        "RX_imu_comp": RX_IMU_compensated
                }