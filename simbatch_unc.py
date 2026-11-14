from digitalshadow.devices.Floater import *
from digitalshadow.devices.IMU_models import load_imu_model, calibrate_floater
from digitalshadow.devices.Transmitter import *
from plot_simulation import *
from simulation import *

TX_LIMIT = 3
HYDROMATE_SIMULATION = True
TRYCATCH = True

if __name__ == '__main__':
        SEEDS = np.arange(1,10)
        STEPS = 3600
        NUMBER_OF_FLOATERS = 5
        ferr = open("seeds_err.txt","w")
        RESURFACE_PERIODS = [120,300,600,900,1800]

        for SEED in SEEDS:      
                for RESS in RESURFACE_PERIODS:
                        SIMULATION_NAME=f'simulation{SEED}_RES_{RESS}_unc'
                        random_sim = np.random.default_rng(SEED)
                        
                        Center = [45.92458622396349,-20.45658040956359]
                        #Center = [20.832813, 88.698390] # India, low depth

                        TX_x = random_sim.uniform(500, 3000) * random_sim.choice([-1, 1])
                        TX_y = random_sim.uniform(500, 3000) * random_sim.choice([-1, 1])

                        TX = Transmitter(ID=-1, gt_x=TX_x, gt_y=TX_y, gt_z=10, dt=1)

                        norm = np.hypot(TX_x, TX_y)
                        ux, uy = -TX_x / norm, -TX_y / norm

                        random_deg = random_sim.uniform(-30, 30)
                        theta = np.deg2rad(random_deg)
                        c, s = np.cos(theta), np.sin(theta)

                        uv_x = ux * c - uy * s
                        uv_y = ux * s + uy * c 


                        TX.set_initial_velocity(5*uv_x,5*uv_y,0)
                        TX.Rho = 0.999
                        TX.set_sigma(0.05,0.05,0)

                        floaters = []
                        for i in range(NUMBER_OF_FLOATERS):
                                f = Floater(ID = i,
                                                gt_x=np.random.uniform(-300,300),
                                                gt_y=np.random.uniform(-300,300),
                                                gt_z=10,
                                                NF=NUMBER_OF_FLOATERS,
                                                dt=1.0,
                                                seed = i+1)

                                f = load_imu_model(f,'ADIS16470',dt=1.0)
                                #f = calibrate_floater(f)
                                f.set_initial_velocity(random_sim.uniform(0,1), random_sim.uniform(0,1), 0.0)
                                f.set_sigma(random_sim.uniform(0,0.01), random_sim.uniform(0,0.01), 0.0)
                                f.Rho = 0.999
                                f.use_compass = True
                                f.TX_LIMIT=3
                                f.PLR = 0.0
                                f.MDS_freq = 900
                                f.RESURFACE_freq = RESS
                                f.incremental_resurface = False
                                floaters.append(f)

                        if TRYCATCH:
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
                        else:
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
