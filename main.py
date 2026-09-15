from Floater import *
from coordinate_generator import geo_to_local
from simulation import *

if __name__ == '__main__':
        NUMBER_OF_FLOATERS = 5
        Center = [20.832813, 88.698390] # India, low depth

        # === Transmitter initialization ===
        TX = Transmitter(ID=-1,
                        gt_x=0,
                        gt_y=0,
                        gt_z=20,
                        dt=1.0)
        TX.set_rho(1.0)
        TX.set_sigma(0.20, 0.20, 0.0)
        TX.set_initial_velocity(0,0,0)
        
        L  = 100

        # === Floater initialization ===
        floaters = []
        for n in range(NUMBER_OF_FLOATERS):            
                # Creation of a new floater
                if n == 0:
                        f = Floater(ID = n,
                                        gt_x=0,
                                        gt_y=L,
                                        gt_z=37,
                                        NF=NUMBER_OF_FLOATERS,
                                        dt=1.0)
                elif n == 1:
                        f = Floater(ID = n,
                                        gt_x=L,
                                        gt_y=0,
                                        gt_z=37,
                                        NF=NUMBER_OF_FLOATERS,
                                        dt=1.0)
                else:
                        f = Floater(ID = n,
                                        gt_x=np.random.uniform()*100,
                                        gt_y=np.random.uniform()*100,
                                        gt_z=37,
                                        NF=NUMBER_OF_FLOATERS,
                                        dt=1.0)
                        

                f.set_initial_velocity(0.2, 0.2, 0.0)
                f.set_rho(0.999)
                f.set_sigma(0.2, 0.2, 0.0)
                floaters.append(f)

        sim = Simulation(transmitter=TX,
                        Floaters=floaters,
                        Steps=200,
                        Center=Center,
                        seed=5)

        sim.run_simulation()