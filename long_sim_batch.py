from digitalshadow.devices.Floater import *
from digitalshadow.devices.IMU_models import load_imu_model, calibrate_floater
from digitalshadow.devices.Transmitter import *
from plot_simulation import *
from simulation import *

TX_LIMIT = 3
HYDROMATE_SIMULATION = True

DE_TI = 10

if __name__ == '__main__':
        SEEDS = np.arange(56,57)
        #STEPS = 360
        RESS = [600,900,1800,3600,7200,10800]
        NUMBER_OF_FLOATERS = 6
        ferr = open("seeds_err.txt","w")

        for SEED in SEEDS:
                for res in RESS:   
                        SIMULATION_NAME=f'long_zim{SEED}_res{res}'
                        random_sim = np.random.default_rng(SEED)
                        
                        Center = [-19.550733287404668,80.08335006633634]
                        #Center = [20.832813, 88.698390] # India, low depth

                        
                        TX_x = random_sim.uniform(500,3000)
                        TX_y = random_sim.uniform(500,3000)
                        q1 = 1 if random_sim.uniform(0,1) > 0.5 else -1
                        q2 = 1 if random_sim.uniform(0,1) > 0.5 else -1


                        floaters = []
                        VINIT_X = random_sim.uniform(-1,-0.5)
                        VINIT_Y = random_sim.uniform(0,1)
                        for i in range(NUMBER_OF_FLOATERS):
                                R = 600
                                f = Floater(ID = i,
                                                gt_x=R*np.cos(np.deg2rad(360*i/NUMBER_OF_FLOATERS)),
                                                gt_y=R*np.sin(np.deg2rad(360*i/NUMBER_OF_FLOATERS)),
                                                gt_z=10,
                                                NF=NUMBER_OF_FLOATERS,
                                                dt=DE_TI,
                                                seed = i*10+100)

                                f = load_imu_model(f,'HG4930',param_seed=i,dt=DE_TI)
                                #f = calibrate_floater(f)

                                f.set_initial_velocity(VINIT_X * random_sim.uniform(0.7,1.3), VINIT_Y * random_sim.uniform(0.7,1.3), 0.0)
                                f.set_sigma(random_sim.uniform(0,0.2), random_sim.uniform(0,0.2), 0.0)
                                f.Rho = 0.999
                                f.use_compass = True
                                f.TX_LIMIT = 3
                                f.PLR = 0.0
                                f.MDS_freq = (300 / DE_TI) # Secondi
                                f.RESURFACE_freq = (res / DE_TI)
                                floaters.append(f)

                        #try:
                        TXX = np.load("TX.npy")*15
                        TXX = TXX - np.asarray([6000,7000])

                        TXX = np.stack((TXX[:,0],TXX[:,1],15*np.ones(TXX.shape[0]))).T
                        sampled_TX = TXX[::3]
                        
                        sim = Simulation(
                                        transmitter_coordinates=sampled_TX,
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
                        '''
                        except Exception as e:
                                print("ERRORe")
                                ferr.write(f"{SEED}\n")
                                continue
                        '''