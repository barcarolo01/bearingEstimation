
import shutil

import numpy as np
from digitalshadow.devices.Floater import *
from digitalshadow.underwater_sim.bellhop_to_wav import *
from hydromate.app_bellhop import AppBellhop

FS_OUT = 96000
C_SOUND = 1540
    
def ping_pair(Floater1_coordiantes,Floater2_coordiantes):
    """
    This funciton takes the coordinates of two floaters and run a BELLHOP simulation to emulate 
    a ranging operation: the delay of the first arrival is taken as the as the propagation delay 
    """
    Floater2_coordiantes[[0,1]] = Floater2_coordiantes[[1,0]]
    Floater1_coordiantes = np.asarray([Floater1_coordiantes[1],Floater1_coordiantes[0],Floater1_coordiantes[2]])

    app = AppBellhop("ping",
                    np.asarray(Floater1_coordiantes),
                    np.asarray(Floater2_coordiantes),
                    "CVWT", "A",
                    [-89,89],
                    10000,
                    nrr = 1,
                    nrd=1,)

    app.set_paths("C:/Users/Nicola/Desktop/TESI/HYDROMATE/hm_code_py/.env")
    app.run_sim()

    arrivals = read_arr('ping_1/ping.arr')
    used = sorted(arrivals, key=lambda a: a[2])   # Sort by REAL delay
    delay_first_arrival_ms = 1000*used[0][2]
    shutil.rmtree(f'ping_1')    

    return delay_first_arrival_ms # Delay, in seconds