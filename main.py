from digitalshadow.devices.Floater import *
from digitalshadow.devices.IMU_models import load_imu_model
from digitalshadow.devices.Transmitter import *
from plot_simulation import plot_sim_result
from simulation import *


TX_LIMIT = 3
HYDROMATE_SIMULATION = True

if __name__ == '__main__':
        SEED = 1002
        SIMULATION_NAME=f'simulation{SEED}'
        random_sim = np.random.default_rng(SEED)
        NUMBER_OF_FLOATERS = 4
        STEPS = 600
        Center = [20.832813, 88.698390] # India, low depth
        
        TX = Transmitter(-1,-100,-100,10,1.0)
        TX.Rho = 0.999
        TX.set_initial_velocity(-6,8,0)
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
                                dt=1.0)

                f = load_imu_model(f,'ADIS16470',dt=1.0)
                f.set_initial_velocity(random_sim.uniform(-3,0), random_sim.uniform(0,4), 0.0)
                f.set_sigma(random_sim.uniform(0.05), random_sim.uniform(0.05), 0.0)
                f.Rho = 0.999
                #f.Rho_yaw = 1.0
                #f.use_compass = True
                #f.accel_bias = 0
                #f.sigma_accel_bias = np.zeros(3)
                #f.MDS_freq = 2
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
                        seed=100,sim_name=SIMULATION_NAME,
                        HYDROMATE_SIMULATION=HYDROMATE_SIMULATION)
        sim.PACKET_LOSS = 0
        sim.run_simulation()

        plot_sim_result(SIMULATION_NAME)