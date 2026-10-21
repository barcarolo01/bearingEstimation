from digitalshadow.devices.Floater import *
from digitalshadow.devices.IMU_models import load_imu_model, calibrate_floater
from digitalshadow.devices.Transmitter import *
from digitalshadow.utils.Envelope import detect_env
from plot_simulation import *
from simulation import *


TX_LIMIT = 3
HYDROMATE_SIMULATION = True

if __name__ == '__main__':
        SEED = 1
        SIMULATION_NAME=f'simulationCIRCLE_3depth_3'
        random_sim = np.random.default_rng(SEED)
        NUMBER_OF_FLOATERS = 3
        STEPS = 1
        Center = [20.832813, 88.698390] # India, low depth
        #Center = [32.839, -34.635] # Atlantic
        #Center =[43.4144393753542,50.10894650149935]
        
        #TX = Transmitter(-1,0,0,10,1.0,seed=SEED)
        #TX.Rho = 0.995
        #TX.set_initial_velocity(-3,6,0)
        #TX.set_sigma(0.2,0.2,0)
        
        TX = compute_TX_circle_trajectory(50,0,350,35,200)

        floaters = []
        R = 58
        for i in range(NUMBER_OF_FLOATERS):
                if i == 0:
                        x = R * np.cos(np.deg2rad(-30))
                        y = R * np.sin(np.deg2rad(-30))
                        z = 50
                if i == 1:
                        x = R * np.cos(np.deg2rad(90))
                        y = R * np.sin(np.deg2rad(90))
                        z = 50
                if i == 2:
                        x = R * np.cos(np.deg2rad(210))
                        y = R * np.sin(np.deg2rad(210))
                        z = 50

                f = Floater(ID = i+1,
                                #gt_x=100,
                                #gt_y=0,
                                gt_x=x,
                                gt_y=y,
                                gt_z=z,
                                NF=NUMBER_OF_FLOATERS,
                                dt=1.0,
                                seed = i+1)

    
                
                floaters.append(f)

        for p in range(TX.shape[0]): #1
                TX[p,2] = 10 + p * 2

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