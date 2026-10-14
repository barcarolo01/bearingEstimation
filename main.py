from digitalshadow.devices.Floater import *
from digitalshadow.devices.IMU_models import load_imu_model, calibrate_floater
from digitalshadow.devices.Transmitter import *
from plot_simulation import plot_sim_result
from simulation import *


TX_LIMIT = 3
HYDROMATE_SIMULATION = True

if __name__ == '__main__':
        SEEDS = np.arange(1,12,1)
        SEEDS = [202]
        MDS = 300
        PLR = 0.0
        for SEED in SEEDS:      
                SIMULATION_NAME=f'simulation{SEED}_MDS{MDS}_PLR{PLR}'
                random_sim = np.random.default_rng(SEED)
                NUMBER_OF_FLOATERS = 4
                STEPS = 900
                Center = [20.832813, 88.698390] # India, low depth
                #Center = [32.839, -34.635] # Atlantic
                
                TX = Transmitter(-1,random_sim.uniform(-1000,0),random_sim.uniform(-1000,0),10,1.0,seed=SEED)
                TX.Rho = 0.999
                TX.set_initial_velocity(-3,8,0)
                TX.set_sigma(0.0,0.1,0)
                
                #TX = compute_TX_circle_trajectory(20,0,350,35,200)

                floaters = []
                for i in range(NUMBER_OF_FLOATERS):
                        f = Floater(ID = i,
                                        gt_x=random_sim.uniform(-1000,1000),
                                        gt_y=random_sim.uniform(-1000,1000),
                                        #gt_x=(-100 if i==1 else 100),
                                        #gt_y=0,
                                        gt_z=10,
                                        NF=NUMBER_OF_FLOATERS,
                                        dt=1.0,
                                        seed = i+1)

                        f = load_imu_model(f,'ADIS16470',dt=1.0)
                        #f = calibrate_floater(f)
                        f.set_initial_velocity(random_sim.uniform(0,3), random_sim.uniform(0,3), 0.0)
                        f.set_sigma(random_sim.uniform(0.01), random_sim.uniform(0.01), 0.0)
                        f.Rho = 0.9999
                        f.use_compass = True
                        f.TX_LIMIT=3
                        f.PLR = PLR
                        f.MDS_freq = MDS
                        
                        floaters.append(f)

                sim = Simulation(
                                #transmitter_coordinates=TX,
                                Floaters=floaters,
                                Steps=STEPS,
                                Center=Center,
                                transmitter=TX,
                                seed=100,
                                sim_name=SIMULATION_NAME,
                                HYDROMATE_SIMULATION=HYDROMATE_SIMULATION)
                sim.PACKET_LOSS = 0
                sim.run_simulation()

                plot_sim_result(SIMULATION_NAME)