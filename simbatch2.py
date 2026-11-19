from digitalshadow.devices.Floater import *
from digitalshadow.devices.IMU_models import load_imu_model, calibrate_floater
from digitalshadow.devices.Transmitter import *
from plot_simulation import *
from simulation import *

TX_LIMIT = 3
HYDROMATE_SIMULATION = True

if __name__ == '__main__':
        SEEDS = np.arange(56,57)
        STEPS = 3600
        RESS = [300,600,900,1200,1800,3600]
        NUMBER_OF_FLOATERS = 6
        ferr = open("seeds_err.txt","w")

        for SEED in SEEDS:
                for res in RESS:   
                        SIMULATION_NAME=f'ring{SEED}_res{res}'
                        random_sim = np.random.default_rng(SEED)
                        
                        #Center = [45.92458622396349,-20.45658040956359]
                        #Center = [20.832813, 88.698390] # India, low depth
                        Center = [13.415765075632192,42.95190985526649] # HORMUTz

                        
                        TX_x = random_sim.uniform(500,3000)
                        TX_y = random_sim.uniform(500,3000)
                        q1 = 1 if random_sim.uniform(0,1) > 0.5 else -1
                        q2 = 1 if random_sim.uniform(0,1) > 0.5 else -1

                        '''
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
                        TX.Rho = 1
                        '''

                        floaters = []
                        VINIT = random_sim.uniform(-1,1)
                        for i in range(NUMBER_OF_FLOATERS):
                                R = 600
                                f = Floater(ID = i,
                                                gt_x=R*np.cos(np.deg2rad(360*i/NUMBER_OF_FLOATERS)),
                                                gt_y=R*np.sin(np.deg2rad(360*i/NUMBER_OF_FLOATERS)),
                                                gt_z=10,
                                                NF=NUMBER_OF_FLOATERS,
                                                dt=1.0,
                                                seed = i*10+100)

                                f = load_imu_model(f,'HG4930',param_seed=i,dt=1.0)
                                #f = calibrate_floater(f)

                                f.set_initial_velocity(VINIT * random_sim.uniform(0.7,1.3), VINIT * random_sim.uniform(0.7,1.3), 0.0)
                                f.set_sigma(random_sim.uniform(0,0.1), random_sim.uniform(0,0.1), 0.0)
                                f.Rho = 0.999
                                f.use_compass = True
                                f.TX_LIMIT = 3
                                f.PLR = 0.0
                                f.MDS_freq = 300
                                f.RESURFACE_freq = 1800
                                f.RESURFACE_freq = res
                                floaters.append(f)

                        try:
                                TXX = np.load("TX.npy")*6
                                TXX = TXX - np.asarray([2400,2400])
                                TXX = np.asarray((TXX[:,0],TXX[:,1],10*np.ones(TXX.shape[0]))).T
                                print(TXX.shape)
                                sim = Simulation(
                                                transmitter_coordinates=TXX,
                                                Floaters=floaters,
                                                #Steps=STEPS,
                                                Center=Center,
                                                #transmitter=TX,
                                                seed=SEED,
                                                sim_name=SIMULATION_NAME,
                                                HYDROMATE_SIMULATION=HYDROMATE_SIMULATION)
                                sim.PACKET_LOSS = 0
                                sim.run_simulation()
                                plot_simulation_results(SIMULATION_NAME)
                        except Exception as e:
                                ferr.write(f"{SEED}\n")
                                continue