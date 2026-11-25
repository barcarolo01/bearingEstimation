from digitalshadow.devices.Floater import *
from digitalshadow.devices.IMU_models import load_imu_model, calibrate_floater
from digitalshadow.devices.Transmitter import *
from plot_simulation import *
from simulation import *
import sys

TX_LIMIT = 3
HYDROMATE_SIMULATION = True

DE_TI = 10

if __name__ == '__main__':
        TXX = np.load("TX2.npy")*10
        TXX = TXX - np.asarray([4000,4000])
        TXX = np.stack((TXX[:,0],TXX[:,1],15*np.ones(TXX.shape[0]))).T
        #print(TXX.shape)
        #exit()
        SEED = int(sys.argv[1])
        res = 10800
        NUMBER_OF_FLOATERS = 5
        ferr = open("seeds_err.txt","w")

        SIMULATION_NAME=f'999_3h_{SEED}'
        not_rand = np.random.default_rng(0)
        random_sim = np.random.default_rng(SEED)
        
        Center = [-19.550733287404668,80.08335006633634]
        #Center = [20.832813, 88.698390] # India, low depth
        
        TX_x = random_sim.uniform(500,3000)
        TX_y = random_sim.uniform(500,3000)
        q1 = 1 if random_sim.uniform(0,1) > 0.5 else -1
        q2 = 1 if random_sim.uniform(0,1) > 0.5 else -1


        floaters = []
        VINIT_X =  0
        VINIT_Y = 0
        for i in range(NUMBER_OF_FLOATERS):
                R = 400
                f = Floater(ID = i,
                                gt_x=R*np.cos(np.deg2rad(360*i/NUMBER_OF_FLOATERS)),
                                gt_y=R*np.sin(np.deg2rad(360*i/NUMBER_OF_FLOATERS)),
                                gt_z=10,
                                NF=NUMBER_OF_FLOATERS,
                                dt=DE_TI,
                                seed = i*10+100)

                f = load_imu_model(f,'HG4930',param_seed=SEED*1000+i,dt=DE_TI)
                #f = calibrate_floater(f)

                f.set_initial_velocity(0,0,0)
                f.set_sigma(random_sim.uniform(0.0,0.1), random_sim.uniform(0.0,0.1), 0.0)
                #f.Rho = 0.98 # PRIMO BATCH
                f.Rho = 0.999
                f.use_compass = True
                f.TX_LIMIT = 3
                f.PLR = 0.0
                f.MDS_freq = (600 / DE_TI) # Secondi
                f.RESURFACE_freq = (res / DE_TI)
                floaters.append(f)

        #try:
        

        #TXX = TXX[::108]
        
        
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
        '''
        except Exception as e:
                print("ERRORe")
                ferr.write(f"{SEED}\n")
                continue
        '''