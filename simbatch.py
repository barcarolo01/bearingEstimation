from digitalshadow.devices.Floater import *
from digitalshadow.devices.IMU_models import load_imu_model, calibrate_floater
from digitalshadow.devices.Transmitter import *
from plot_simulation import *
from simulation import *

TX_LIMIT = 3
HYDROMATE_SIMULATION = True

if __name__ == '__main__':
        SEEDS = np.arange(1,100)
        STEPS = 3600
        NUMBER_OF_FLOATERS = 5
        ferr = open("seeds_err.txt","w")

        for SEED in SEEDS:      
                SIMULATION_NAME=f'simulation{SEED}'
                random_sim = np.random.default_rng(SEED)
                
                Center = [45.92458622396349,-20.45658040956359]
                #Center = [20.832813, 88.698390] # India, low depth

                
                TX_x = random_sim.uniform(500,3000)
                TX_y = random_sim.uniform(500,3000)
                q1 = 1 if random_sim.uniform(0,1) > 0.5 else -1
                q2 = 1 if random_sim.uniform(0,1) > 0.5 else -1
                
                TX = Transmitter(ID=-1,
                                 gt_x=TX_x*q1,
                                 gt_y=TX_y*q2,
                                 gt_z=10,
                                 dt=1)

                random_deviation = 0.33
                random_deg = random_sim.normal()
                print(random_deg)

                uv_x = ((TX_x*q1) / np.sqrt((TX_x*q1)**2 + (TX_y*q2)**2))
                uv_y = ((TX_y*q1) / np.sqrt((TX_x*q1)**2 + (TX_y*q2)**2))
                uv_x = uv_x * np.cos(np.deg2rad(random_deg))
                uv_y = uv_y * np.sin(np.deg2rad(random_deg))


                TX.set_initial_velocity(-5*uv_x,5*uv_y,0)
                TX.Rho = 0.999
                TX.set_sigma(0.05,0.05,0)

                floaters = []
                for i in range(NUMBER_OF_FLOATERS):
                        '''
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
                        '''
                
                        f = Floater(ID = i,
                                        gt_x=np.random.uniform(-300,300),
                                        gt_y=np.random.uniform(-300,300),
                                        gt_z=10,
                                        NF=NUMBER_OF_FLOATERS,
                                        dt=1.0,
                                        seed = i+1)

                        f = load_imu_model(f,'ADIS16470',dt=1.0)
                        f = calibrate_floater(f)
                        f.set_initial_velocity(random_sim.uniform(0,1), random_sim.uniform(0,1), 0.0)
                        f.set_sigma(random_sim.uniform(0,0.01), random_sim.uniform(0,0.01), 0.0)
                        f.Rho = 0.999
                        f.use_compass = True
                        f.TX_LIMIT=3
                        f.PLR = 0.0
                        f.MDS_freq = 900
                        f.RESURFACE_freq = 1800
                        floaters.append(f)

                try:
                        sim = Simulation(
                                        #transmitter_coordinates=TX,
                                        Floaters=floaters,
                                        Steps=STEPS,
                                        Center=Center,
                                        transmitter=TX,
                                        seed=SEED,
                                        sim_name=SIMULATION_NAME,
                                        HYDROMATE_SIMULATION=HYDROMATE_SIMULATION)
                        sim.PACKET_LOSS = 0
                        sim.run_simulation()
                        plot_simulation_results(SIMULATION_NAME)
                except Exception as e:
                        ferr.write(f"{SEED}\n")
                        continue