import os
import shutil
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from coordinate_generator import *
from discrete_hydromate_single import run_discrete_hydromate_single
from filter_trajectory import *
from point_estimation import *
from maps.build_local_map import build_local_cartesian_map
from maps.map_common import Track
from floater_ping import *
from utils_runner import *
from Floater import *

FONTSIZE = 18
TX_LIMIT = 2

ADD_PSI_ERROR = False
HYDROMATE_SIMULATION = False

colors = {
                'imu':  "#0000FF",
                'imu_mds':  "#2AB040",
                'imu_rev':  "#FC03D3",
                'imu_mds_rev':  "#D6940F",
                'imu_compensated': "#FF1111"
         }


class Simulation:
        def __init__(self, transmitter, Floaters, Steps, Center, seed=256123):
                load_dotenv()
                np.random.seed(seed)
                self.NUMBER_OF_FLOATERS = len(Floaters)
                self.Center = Center
                self.transmitter = transmitter
                self.Floaters = Floaters
                self.SIMULATION_STEPS = Steps
                self.NUMBER_OF_HYDROPHONES = int(os.getenv('NUMBER_OF_HYDROPHONES'))
                self.SAMPLING_FREQUENCY = int(os.getenv('SAMPLING_FREQUENCY'))

                # Floater data for vessel DoA estimation
                self.bearing_arrays = np.full((self.NUMBER_OF_FLOATERS,self.SIMULATION_STEPS), np.nan)
                self.elevation_arrays = np.full((self.NUMBER_OF_FLOATERS,self.SIMULATION_STEPS), np.nan)
                self.psi_error_arrays = np.full((self.NUMBER_OF_FLOATERS,self.SIMULATION_STEPS), np.nan)

        
        def run_simulation(self):
                TX_Coordinates = np.zeros((self.SIMULATION_STEPS,3))
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
                RX_gt_Coordinates = np.zeros((self.SIMULATION_STEPS,self.NUMBER_OF_FLOATERS,3))

                if os.path.isdir("Synth"):
                        shutil.rmtree("Synth")
                os.makedirs("Synth")

                round_active  = False
                round_order   = []     # Floater transmitting sequence (2N-1 long)
                round_step    = 0
                round_id      = 0

                mustTX = [False] * self.NUMBER_OF_FLOATERS
                for i in range(self.SIMULATION_STEPS):
                        TX_Coordinates[i,:] = local_to_geo(self.Center,self.transmitter.gt_pos)        

                        for n in range(self.NUMBER_OF_FLOATERS):
                                print(f"# Simulation step {i+1}/{self.SIMULATION_STEPS}, Floater {n} #")
                                RX_gt_Coordinates[i,n,:] = local_to_geo(self.Center,self.Floaters[n].gt_pos)

                                # Hydromate simulation
                                if HYDROMATE_SIMULATION:
                                        run_discrete_hydromate_single(TX_Coordinates[i,0],
                                                                TX_Coordinates[i,1],
                                                                TX_Coordinates[i,2],
                                                                RX_gt_Coordinates[i,n,0],
                                                                RX_gt_Coordinates[i,n,1],
                                                                RX_gt_Coordinates[i,n,2],
                                                                (n+1))

                                        # Save as wav segment
                                        for j in range(self.NUMBER_OF_HYDROPHONES):
                                                hydrophone_track = np.load(os.path.join('TMP',f'H{j+1}.npy'))
                                                wav.write(f'Synth/T{i}_F{n+1}_H{j+1}.wav', self.SAMPLING_FREQUENCY, hydrophone_track)
                                                                
                                        self.bearing_arrays[n,i], self.elevation_arrays[n,i]  = compute_single_bearing_angle_complete(timestamp=i, wav_folder='Synth',F_index=(n + 1))

                        # Current shapshot
                        positions = np.array([f.gt_pos for f in self.Floaters])
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
                                for k in range(TX_LIMIT):
                                        if round_step >= len(round_order):
                                                break
                                        tx   = round_order[round_step] # ID of the scheduled transmitter
                                        t_tx = clocks[tx] + math.ceil(1000/TX_LIMIT)
                                        print(f"SIMSTEP {i}, transmission of {tx}")
                                        payload = self.Floaters[tx].on_transmit(t_tx, tx)

                                        self.Floaters[tx].events.append({
                                                "t_local": t_tx, "type": "TX",
                                                "src": self.Floaters[tx].ID, "payload": payload,
                                        })


                                        # Register the reception on the local event log of all other floaters
                                        for m in range(self.NUMBER_OF_FLOATERS):
                                                if m == tx:
                                                        continue

                                                #d = ping_pair(local_to_geo(self.Center,positions[tx]),local_to_geo(self.Center,positions[m]))
                                                d = np.linalg.norm(positions[tx]-positions[m])

                                                t_rx = clocks[m] + d + math.ceil(1000/TX_LIMIT)
                                                self.Floaters[m].on_receive(t_rx, tx, m, payload, self.Floaters[tx].ID)
                                                
                                        round_step += 1
                                if round_step == len(round_order):
                                        round_active = False
                                        print(f"SIMSTEP {i}, ended rangeing")

                        # Advance the simulation for the next step
                        mustTX = [False] * self.NUMBER_OF_FLOATERS
                        for n in range(self.NUMBER_OF_FLOATERS):
                                # At the end of the simulation, the floater emerges
                                if i != (self.SIMULATION_STEPS-1):
                                        mustTX[n] = self.Floaters[n].move()
                                        
                        self.transmitter.move() # Vessel motion

                        
     
                names   = ['imu', 'imu_mds', 'imu_rev', 'imu_mds_rev', 'imu_compensated']
                results = [] 

                for n in range(self.NUMBER_OF_FLOATERS):
                        res = self.Floaters[n].return_results(fuse=False)

                        self.psi_error_arrays[n,:] = res['psi_err'].copy()
                        results.append(res)

                        gt = res['gt']
                        GT[:, n, :] = gt

                        fig, ax = plt.subplots(figsize=(9, 5))

                        for name in names:
                                pos = res[name]
                                targets[name][:, n, :] = pos
                                ax.plot(np.linalg.norm(pos - gt, axis=1), color=colors[name], lw=1.8, label=name)

                        ax.set_title("Positioning error vs ground truth")
                        ax.set_xlabel("Simulation steps")
                        ax.set_ylabel("Metri")
                        ax.grid(True, ls='--', alpha=.5)
                        for k in self.Floaters[n].Resurface_index:
                                if 1 <= k <= self.SIMULATION_STEPS:
                                        ax.axvline(k - 1, color='gray', ls=':', lw=.8)
                        ax.legend(loc="upper right", frameon=True)

                        fig.suptitle(f"Floater {n}", fontsize=14)
                        fig.tight_layout()
                        plt.savefig(f"floater_{n}_errors.png", dpi=120)
                        plt.close(fig)

                RX_fw_IMU          = local_to_geo(self.Center, RX_fw_IMU)
                RX_fw_IMU_MDS      = local_to_geo(self.Center, RX_fw_IMU_MDS)
                RX_bw_IMU          = local_to_geo(self.Center, RX_bw_IMU)
                RX_bw_IMU_MDS      = local_to_geo(self.Center, RX_bw_IMU_MDS)
                RX_IMU_compensated = local_to_geo(self.Center, RX_IMU_compensated)

                print(f"{'Version':22s} {'RMSE':>10s}")
                for name in names:
                        e_all = []
                        for n in range(self.NUMBER_OF_FLOATERS):
                                res = results[n]
                                pos = res[name][1:]
                                e_all.append(np.linalg.norm(pos - res['gt'][1:], axis=1))
                        e = np.concatenate(e_all)
                        print(f"{name:22s} {np.sqrt((e**2).mean()):10.3f}")

                clean_temporary_files()
                if ADD_PSI_ERROR and HYDROMATE_SIMULATION:
                        self.bearing_arrays = wrap_degrees(self.bearing_arrays + self.psi_error_arrays)

                estimated_points = find_points(RX_gt_Coordinates,self.bearing_arrays,self.elevation_arrays)
 
                # Plotting points on the map                
                build_local_cartesian_map(
                        floater_coordinates=RX_gt_Coordinates,
                        TX_coordinates=TX_Coordinates,
                        estimated_vessel_coordinates=estimated_points,
                        tracks=[
                                Track("RX IMU", RX_fw_IMU, "#0000FF"),
                                Track("RX IMU+MDS", RX_fw_IMU_MDS, "#2AB040"),
                                Track("Compensated", RX_bw_IMU, "#FF8822"),
                                ],
                        output_file="maps/local_map.png"
                )

                rmse_flat, rmse_depth = compute_RMSE(TX_Coordinates,estimated_points)
                print(f"RMSE estimated_points:\t {rmse_flat:.1f}")
                print(f"RMSE depth:\t\t {rmse_depth:.1f}")