
import shutil

import numpy as np
from digitalshadow.devices.Floater import *
from digitalshadow.underwater_sim.bellhop_to_wav import build_ir, read_arr
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
                    nrd=1)

    app.set_paths("C:/Users/Nicola/Desktop/TESI/HYDROMATE/hm_code_py/.env")
    app.run_sim()

    rr_values, rd_values, arrivals = read_arr('ping_1/ping.arr')
    h, used = build_ir(arrivals, rd_values, max(rr_values), FS_OUT, n_arrivals=0)
    delay_first_arrival_samples = np.nonzero(h)[0][0]
    delay_first_arrival_ms = 1000*(delay_first_arrival_samples / FS_OUT)
    shutil.rmtree(f'ping_1')    

    return delay_first_arrival_ms # Delay, in seconds