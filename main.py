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
        STEPS = 5
        NUMBER_OF_FLOATERS = 5
        MDS = 300
        PLR = 0.0
        for SEED in SEEDS:      
                SIMULATION_NAME=f'simulation0'
                random_sim = np.random.default_rng(SEED)
                
                Center = [45.92458622396349,-20.45658040956359]
                #Center = [20.832813, 88.698390] # India, low depth

                ''' 
                TX = Transmitter(ID=-1,
                                 gt_x=-50,
                                 gt_y=0,
                                 gt_z=10,
                                 dt=1)
                TX.set_initial_velocity(10,0,0)
                TX.Rho = 1
                '''
                line1 = np.arange(-400,100,10)
                TX_1 = np.asarray([line1,-100*np.ones(line1.shape[0]),10*np.ones(line1.shape[0])]).T

                curve = compute_TX_circle_trajectory(10,180,90,10,100,False)
                curve[:,0] +=  100

                line2 = np.arange(0,400,10)
                TX_2 = np.asarray([200*np.ones(line2.shape[0]),line2,10*np.ones(line2.shape[0])]).T

                TX = np.concat([TX_1,curve,TX_2])

                floaters = []
                for i in range(NUMBER_OF_FLOATERS):
                        if i == 0:
                                x = 0
                                y = 0
                        if i == 1:
                                x = 300
                                y = 100
                        if i == 2:
                                x = -1600
                                y = 0
                        if i == 3:
                                x = -1300
                                y = 100

                
                        f = Floater(ID = i,
                                        gt_x=x,
                                        gt_y=y,
                                        gt_z=10,
                                        NF=NUMBER_OF_FLOATERS,
                                        dt=1.0,
                                        seed = i+1)

                        f = load_imu_model(f,'ADIS16470',dt=1.0)
                        #f = calibrate_floater(f)
                        f.set_initial_velocity(random_sim.uniform(0,0), random_sim.uniform(0,0), 0.0)
                        f.set_sigma(random_sim.uniform(0.01), random_sim.uniform(0.01), 0.0)
                        f.Rho = 1 
                        f.use_compass = True
                        f.TX_LIMIT=3
                        #f.PLR = PLR
                        #f.MDS_freq = MDS
                        
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