from digitalshadow.devices.Floater import *
from digitalshadow.devices.Transmitter import *
from simulation import *

if __name__ == '__main__':
        NUMBER_OF_FLOATERS = 3
        STEPS = 100
        #Center = [20.832813, 88.698390] # India, low depth
        Center = [32.839, -34.635] 


        # === Floater initialization ===
        floaters = []
        f = Floater(ID = 1,
                        gt_x=-86.6,
                        gt_y=-50,
                        gt_z=10,
                        NF=NUMBER_OF_FLOATERS,
                        dt=1.0)

        f.set_initial_velocity(0.0, 0.0, 0.0)
        f.sigma_yaw_rate = 0
        f.gt_omega = 0
        f.use_compass = False
        f.Rho = 1
        f.Rho_yaw = 1
        f.set_sigma(0.0, 0.0, 0.0)
        floaters.append(f)

        f = Floater(ID = 2,
                gt_x=86.6,
                gt_y=-50,
                gt_z=10,
                NF=NUMBER_OF_FLOATERS,
                dt=1.0)

        f.set_initial_velocity(0.0, 0.0, 0.0)
        f.sigma_yaw_rate = 0
        f.gt_omega = 0
        f.Rho = 1
        f.Rho_yaw = 1
        f.set_sigma(0.0, 0.0, 0.0)
        f.use_compass = False
        floaters.append(f)

        f = Floater(ID = 3,
                gt_x=0,
                gt_y=100,
                gt_z=10,
                NF=NUMBER_OF_FLOATERS,
                dt=1.0)

        f.set_initial_velocity(0.0, 0.0, 0.0)
        f.sigma_yaw_rate = 0
        f.gt_omega = 0
        f.Rho = 1
        f.Rho_yaw = 1
        f.set_sigma(0.0, 0.0, 0.0)
        f.use_compass = False
        floaters.append(f)
        

        sim = Simulation(Floaters=floaters,
                        #Steps=STEPS,
                        Center=Center,
                        transmitter_coordinates=compute_TX_circle_trajectory(10,0,350,35,200),
                        seed=5)

        sim.run_simulation()