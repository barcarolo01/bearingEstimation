from digitalshadow.devices.Floater import *
from digitalshadow.devices.IMU_models import load_imu_model, calibrate_floater
from digitalshadow.devices.Transmitter import *
from digitalshadow.utils.Envelope import detect_env
from plot_simulation import *
from simulation import *


TX_LIMIT = 3
HYDROMATE_SIMULATION = True

if __name__ == '__main__':
        SEED = 2
        SIMULATION_NAME=f'simulationCIRCLE'
        random_sim = np.random.default_rng(SEED)
        NUMBER_OF_FLOATERS = 2
        STEPS = 1
        #Center = [20.832813, 88.698390] # India, low depth
        #Center = [32.839, -34.635] # Atlantic
        Center =[43.4144393753542,50.10894650149935]
        
        #TX = Transmitter(-1,0,0,10,1.0,seed=SEED)
        #TX.Rho = 0.995
        #TX.set_initial_velocity(-3,6,0)
        #TX.set_sigma(0.2,0.2,0)
        
        TX = compute_TX_circle_trajectory(10,0,359,359,200)

        floaters = []
        for i in range(NUMBER_OF_FLOATERS):
                f = Floater(ID = i,
                                #gt_x=100,
                                #gt_y=0,
                                gt_x=(-100 if i==1 else 100),
                                gt_y=0,
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