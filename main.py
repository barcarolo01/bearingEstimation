from digitalshadow.devices.Floater import *
from digitalshadow.devices.IMU_models import load_imu_model
from digitalshadow.devices.Transmitter import *
from simulation import *

TX_LIMIT = 3

if __name__ == '__main__':
        NUMBER_OF_FLOATERS = 10
        STEPS = 1000
        Center = [20.832813, 88.698390] # India, low depth
        #Center = [32.839, -34.635] 
        
        
        TX = Transmitter(-1,-0,0,10,1.0)
        TX.Rho = 0.999
        TX.set_initial_velocity(20,0,0)
        TX.set_sigma(0.3,0.0,0)

        '''
        TX = compute_TX_circle_trajectory(10,0,350,35,200)
        '''

        random_sim = np.random.default_rng(256123)
        floaters = []
        for i in range(NUMBER_OF_FLOATERS):
                f = Floater(ID = i,
                                gt_x=random_sim.uniform(-500,500),
                                gt_y=random_sim.uniform(-500,500),
                                #gt_x=(-100 if i==1 else 100),
                                #gt_y=0,
                                gt_z=10,
                                NF=NUMBER_OF_FLOATERS,
                                dt=1.0)

                f = load_imu_model(f,'ADIS16470',dt=1.0)
                f.set_initial_velocity(random_sim.uniform(-0.5,0.5), random_sim.uniform(-0.5,0.5), 0.0)
                f.set_sigma(random_sim.uniform(0.05), random_sim.uniform(0.05), 0.0)
                f.Rho = 0.9999 + random_sim.uniform(0,0.0001)
                f.Rho_yaw = 1.0
                f.use_compass = True
                f.accel_bias = 0
                f.sigma_accel_bias = np.zeros(3)
                f.MDS_freq = 200
                floaters.append(f)

   
        sim = Simulation(
                        #transmitter_coordinates=TX,
                        Floaters=floaters,
                        Steps=STEPS,
                        Center=Center,
                        transmitter=TX,
                        seed=100)

        sim.PACKET_LOSS = 0.1

        sim.run_simulation()