from Floater import *
from coordinate_generator import geo_to_local
from simulation import *

if __name__ == '__main__':
    NUMBER_OF_FLOATERS = 3
    Center = [20.832813, 88.698390] # India, low depth

    # === Transmitter initialization ===
    TX = Transmitter(ID=-1,
                     gt_x=0,
                     gt_y=0,
                     gt_z=0,
                     dt=1.0)
    TX.set_rho(1.0)
    TX.set_sigma(0.00, 0.00, 0.0)
    TX.set_initial_velocity(0.0,0.0,0)

    # === Floater initialization ===
    floaters = []
    for n in range(NUMBER_OF_FLOATERS):            
            # Creation of a new floater
            f = Floater(ID = n,
                        gt_x=(-1)**n*100+np.random.uniform()*100,
                        gt_y=(-1)**n*100+np.random.uniform()*100,
                        gt_z=37,
                        NF=NUMBER_OF_FLOATERS,
                        dt=1.0)

            f.set_initial_velocity(0.0, 0.0, 0.0)
            f.set_rho(1.0)
            f.set_sigma(0.1, 0.1, 0.0)
            floaters.append(f)

    sim = Simulation(transmitter=TX,
                     Floaters=floaters,
                     Steps=50,
                     Center=Center,
                     seed=10)

    sim.run_simulation()