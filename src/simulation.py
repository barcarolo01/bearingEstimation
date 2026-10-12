import os
import shutil
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from digitalshadow.Positioning.coordinate_generator import *
from digitalshadow.underwater_sim.discrete_hydromate_single import *
from digitalshadow.Positioning.point_estimation import *
from digitalshadow.underwater_sim.floater_ping import *
from digitalshadow.utils.utils import *
from digitalshadow.devices.Floater import *
from digitalshadow.Positioning.kalman_filter import *
from digitalshadow.Positioning.tracking_plots import *
from plot_simulation import plot_sim_interactive

FONTSIZE = 18

ADD_PSI_ERROR = False

class Simulation:
        def __init__(self, Floaters, Center, transmitter=None,  Steps=1, transmitter_coordinates=None, sim_name="simulation0", TX_LIMIT=1, seed=256123, HYDROMATE_SIMULATION=True):
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
                self.init_seed = seed
                self.rnd_packetloss = np.random.default_rng(seed+1)
                self.HYDROMATE_SIMULATION = HYDROMATE_SIMULATION

                self.sim_name = sim_name
                if os.path.isdir(self.sim_name):
                        shutil.rmtree(self.sim_name)
                os.makedirs(self.sim_name)
                os.makedirs(os.path.join(self.sim_name,'SynthTracks'))
                
                self.NUMBER_OF_FLOATERS = len(Floaters)
                self.Center = Center
                self.Floaters = Floaters
                self.TX_LIMIT = TX_LIMIT
                self.PACKET_LOSS = 0.0

                for f in Floaters:
                        f.TX_LIMIT = TX_LIMIT

                self.NUMBER_OF_HYDROPHONES = int(os.getenv('NUMBER_OF_HYDROPHONES'))

                if self.NUMBER_OF_HYDROPHONES == 3:
                        precompute_bearing_angles_triangle(0.3)
                elif self.NUMBER_OF_HYDROPHONES == 4:
                        precompute_bearing_angles_square(0.228 / math.sqrt(2))
                elif self.NUMBER_OF_HYDROPHONES == 5:
                        precompute_bearing_angles_complete(0.228 / math.sqrt(2))
                else:
                        raise ValueError(f"Unexpected number of hydrophones ({self.NUMBER_OF_HYDROPHONES}).")


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

        
        def run_simulation(self):
                RX_fw_IMU     = np.zeros((self.SIMULATION_STEPS, self.NUMBER_OF_FLOATERS, 3))
                RX_fw_IMU_MDS = np.zeros((self.SIMULATION_STEPS, self.NUMBER_OF_FLOATERS, 3))
                RX_bw_IMU     = np.zeros((self.SIMULATION_STEPS, self.NUMBER_OF_FLOATERS, 3))
                RX_bw_IMU_MDS = np.zeros((self.SIMULATION_STEPS, self.NUMBER_OF_FLOATERS, 3))
                RX_IMU_compensated = np.zeros((self.SIMULATION_STEPS, self.NUMBER_OF_FLOATERS, 3))


                GT = np.zeros((self.SIMULATION_STEPS, self.NUMBER_OF_FLOATERS, 3))
                
                round_active  = False
                round_order   = []     # Floater transmitting sequence (2N-1 long)
                round_step    = 0
                round_id      = 0

                mustTX = [False] * self.NUMBER_OF_FLOATERS
                for i in range(self.SIMULATION_STEPS):
                        print(f"# Simulation step {i+1}/{self.SIMULATION_STEPS}")
                        if self.transmitter is not None:
                                self.TX_positions[i,:] = self.transmitter.gt_pos

                        TX_Coordinates = local_to_geo(self.Center,self.transmitter.gt_pos)

                        for n in range(self.NUMBER_OF_FLOATERS):
                                self.RX_positions[i,n,:] = self.Floaters[n].gt_pos.copy()
                
                                RX_gt_Coordinates = local_to_geo(self.Center,self.RX_positions[i,n,:])

                                if self.HYDROMATE_SIMULATION:
                                        run_discrete_hydromate_single(  self.sim_name,
                                                                        TX_Coordinates[0],
                                                                        TX_Coordinates[1],
                                                                        TX_Coordinates[2],
                                                                        RX_gt_Coordinates[0],
                                                                        RX_gt_Coordinates[1],
                                                                        RX_gt_Coordinates[2],
                                                                        
                                                                        PSI_RX=self.Floaters[n].gt_psi,
                                                                        seed = self.init_seed * 15 + 2)
                                        
                                        # By convolution, obtain a signal for each of the arrival files
                                        from_arr_to_wav(TX_POS=np.asarray([TX_Coordinates[0],TX_Coordinates[1]]),
                                                        number_mic=NUMBER_OF_HYDROPHONES,
                                                        sim_name = self.sim_name,
                                                        sim_timestamp = i,
                                                        floater_number = (n + 1),
                                                        n_arrivals=0,
                                                        sl_db=160,
                                                        add_noise=False,
                                                        noise_seed=self.init_seed)

                                        if NUMBER_OF_HYDROPHONES == 3:
                                                self.bearing_arrays[n,i], self.elevation_arrays[n,i]  = compute_single_bearing_angle_triangle(track_folder=os.path.join(self.sim_name,'SynthTracks'), timestamp=i, F_index=(n + 1))
                                        if NUMBER_OF_HYDROPHONES == 4:
                                                self.bearing_arrays[n,i], self.elevation_arrays[n,i]  = compute_single_bearing_angle_square(timestamp=i, wav_folder=os.path.join(self.sim_name,'SynthTracks'),F_index=(n + 1))
                                        if NUMBER_OF_HYDROPHONES == 5:
                                                self.bearing_arrays[n,i], self.elevation_arrays[n,i]  = compute_single_bearing_angle_complete(timestamp=i, wav_folder=os.path.join(self.sim_name,'SynthTracks'),F_index=(n + 1))


                                        
                        # Current clock shapshot
                        clocks    = np.array([f.clk    for f in self.Floaters])

                        # If the network is not already performing a ranging operation
                        if not round_active:
                                triggers = []

                                for n in range(self.NUMBER_OF_FLOATERS):
                                        if mustTX[n]:
                                                triggers.append(n)

                                if triggers:
                                        print(f"[T {i}] TRIGGER MDS round {round_id+1}")
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

                                        print(f"[T {i}] Ranging round nr. {round_id} triggered by floater {self.Floaters[t].ID}")

        
                        # If a ranging round is ongoing
                        if round_active:
                                for k in range(self.TX_LIMIT):
                                        if round_step >= len(round_order):
                                                break
                                        tx   = round_order[round_step] # ID of the scheduled transmitter
                                        t_tx = clocks[tx] + math.ceil(1000/self.TX_LIMIT)
                                        print(f"[T {i}] Transmission of floater {tx}")
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

                                                if self.rnd_packetloss.uniform(0,1) >= self.PACKET_LOSS:
                                                        #delay_ms = ping_pair(local_to_geo(self.Center,self.Floaters[tx].gt_pos),local_to_geo(self.Center,self.Floaters[m].gt_pos))
                                                        delay_ms = 1000* np.linalg.norm(self.Floaters[tx].gt_pos[:2]-self.Floaters[m].gt_pos[:2]) / 1500
                                                        t_rx = clocks[m] + delay_ms + math.ceil(1000/self.TX_LIMIT)
                                                        self.Floaters[m].on_receive(t_rx, tx, m, payload, self.Floaters[tx].ID)
                                                else:
                                                        lost+=1
                                                tot+=1
                                            
                                        round_step += 1
                                if round_step == len(round_order):
                                        round_active = False
                                       

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

                
                psi_error_arrays = np.zeros((self.SIMULATION_STEPS,self.NUMBER_OF_FLOATERS))
                for n in range(self.NUMBER_OF_FLOATERS):
                        floater_results = self.Floaters[n].return_results(fuse=False)
                        RX_fw_IMU[:,n,:] = floater_results['imu'].copy()
                        RX_fw_IMU_MDS[:,n,:] = floater_results['imu_mds'].copy()
                        RX_IMU_compensated[:,n,:] = floater_results['imu_compensated'].copy()
                        psi_error_arrays[:,n] = floater_results['psi_err'].copy()


                bearings_with_psi_error = np.full(self.bearing_arrays.shape,np.nan)
                if ADD_PSI_ERROR:
                        bearings_with_psi_error = wrap_degrees(self.bearing_arrays + psi_error_arrays)
                
                np.savez(os.path.join(self.sim_name,"results.npz"),
                        Center_coordinates = self.Center,
                        NUMBER_OF_FLOATERS = self.NUMBER_OF_FLOATERS,
                        SIMULATION_STEPS = self.SIMULATION_STEPS,
                        RX = self.RX_positions,
                        RX_IMU = RX_fw_IMU,
                        RX_IMU_MDS = RX_fw_IMU_MDS,
                        RX_IMU_comp = RX_IMU_compensated,
                        TX = self.TX_positions,
                        bearings_no_psi_error = self.bearing_arrays,
                        bearings_with_psi_error = bearings_with_psi_error,
                        elevations = self.elevation_arrays)

                clean_temporary_files(self.sim_name)
                print("End of simulation.")

                #plot_sim_interactive(self.sim_name)

                return {
                        "Center_coordinates":self.Center,
                        "RX_gt_pos": self.RX_positions,
                        "TX_gt_pos": self.TX_positions,
                        "RX_imu": RX_fw_IMU,
                        "RX_imu_mds": RX_fw_IMU_MDS,
                        "RX_imu_comp": RX_IMU_compensated,
                        'bearing_arrays':self.bearing_arrays
                }