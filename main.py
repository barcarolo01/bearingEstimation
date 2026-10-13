from digitalshadow.devices.Floater import *
from digitalshadow.devices.IMU_models import load_imu_model, calibrate_floater
from digitalshadow.devices.Transmitter import *
from plot_simulation import plot_sim_result
from simulation import *


TX_LIMIT = 3
HYDROMATE_SIMULATION = True

if __name__ == '__main__':
        SEEDS = np.arange(1,12,1)
        SEEDS = [12]
        for SEED in SEEDS:
                SIMULATION_NAME=f'simulation{SEED}'
                random_sim = np.random.default_rng(SEED)
                NUMBER_OF_FLOATERS = 4
                STEPS = 600
                Center = [20.832813, 88.698390] # India, low depth
                #Center = [32.839, -34.635] # Atlantic
                
                TX = Transmitter(-1,random_sim.uniform(-500,500),random_sim.uniform(-500,500),10,1.0,seed=SEED)
                TX.Rho = 0.995
                TX.set_initial_velocity(-3,6,0)
                TX.set_sigma(0.2,0.2,0)
                
                #TX = compute_TX_circle_trajectory(20,0,350,35,200)

                floaters = []
                for i in range(NUMBER_OF_FLOATERS):
                        f = Floater(ID = i,
                                        gt_x=random_sim.uniform(-300,300),
                                        gt_y=random_sim.uniform(-300,300),
                                        #gt_x=(-100 if i==1 else 100),
                                        #gt_y=0,
                                        gt_z=10,
                                        NF=NUMBER_OF_FLOATERS,
                                        dt=1.0,
                                        seed = i+1)

                        f = load_imu_model(f,'ADIS16470',dt=1.0)
                        f = calibrate_floater(f)
                        f.set_initial_velocity(random_sim.uniform(-3,3), random_sim.uniform(-3,3), 0.0)
                        f.set_sigma(random_sim.uniform(0.05), random_sim.uniform(0.05), 0.0)
                        f.Rho = 0.999
                        f.use_compass = True
                        f.use_compass = False
                        f.compass_bias = 0
                        f.compass_bias_sigma = 0
                        f.TX_LIMIT=3
                        
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