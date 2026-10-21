from digitalshadow.devices.Floater import *
from digitalshadow.devices.IMU_models import load_imu_model, calibrate_floater
from digitalshadow.devices.Transmitter import *
from plot_simulation import *
from simulation import *


TX_LIMIT = 3
HYDROMATE_SIMULATION = True

if __name__ == '__main__':
        SEEDS = np.arange(1,12,1)
        SEEDS = [501]
        STEPS = 1
        NUMBER_OF_FLOATERS = 2
        MDS = 300
        PLR = 0.0
        for SEED in SEEDS:      
                SIMULATION_NAME=f'simulationGarda'
                random_sim = np.random.default_rng(SEED)
                
                Center = [29.809284451457938,49.27237481798266]
                #Center = [20.832813, 88.698390] # India, low depth

                TX = compute_TX_circle_trajectory(10,90,270,37,200)

                floaters = []
                for i in range(NUMBER_OF_FLOATERS):
                        if i == 0:
                                x = 0
                                y = 0
                        if i == 1:
                                x = 0
                                y = 1000
                
                        f = Floater(ID = i+1,
                                        gt_x=x,
                                        gt_y=y,
                                        gt_z=10,
                                        NF=NUMBER_OF_FLOATERS,
                                        dt=1.0,
                                        seed = i+1)

                        floaters.append(f)

                sim = Simulation(
                                transmitter_coordinates=TX,
                                Floaters=floaters,
                                #Steps=STEPS,
                                Center=Center,
                                #transmitter=TX,
                                seed=100,
                                sim_name=SIMULATION_NAME,
                                HYDROMATE_SIMULATION=HYDROMATE_SIMULATION)
                sim.PACKET_LOSS = 0
                sim.run_simulation()

                plot_simulation_results(SIMULATION_NAME)