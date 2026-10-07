import os
import numpy as np
from dotenv import load_dotenv
from hydromate.app_bellhop import AppBellhop
from digitalshadow.devices.floater_geometry import get_hydrophones_coordinates

load_dotenv()
NUMBER_OF_HYDROPHONES = int(os.getenv('NUMBER_OF_HYDROPHONES'))
HYDROMATE_PY_PATH = os.getenv('HYDROMATE_PY_PATH')

def run_discrete_hydromate_single(sim_name, Lat_TX, Lon_TX, depth_TX, Lat_RX, Lon_RX, depth_RX, PSI_RX = 0,seed=1):
    # NUMBER_OF_HYDROPHONES check
    if NUMBER_OF_HYDROPHONES not in [3,4,5]:
        raise ValueError(f"NUMBER_OF_HYDROPHONES bust be either 3, 4 or 5 (while it is {NUMBER_OF_HYDROPHONES}).")

    Floater_hydrophones = get_hydrophones_coordinates(Lat_RX,Lon_RX,depth_RX,NUMBER_OF_HYDROPHONES, PSI_RX)
    
    # Swap latitude and longitude (for HYDROMATE compatibility)
    Floater_hydrophones[:,[0,1]] = Floater_hydrophones[:,[1,0]]

    TX_position = np.asarray([Lon_TX,Lat_TX,depth_TX])

    app = AppBellhop(f"HM_out_{sim_name}",
                    np.asarray(TX_position),
                    np.asarray(Floater_hydrophones),
                    "CVWT", "A",
                    [-89,89],
                    10000,
                    nrr=1,
                    nrd=1)

    app.set_paths(os.path.join(HYDROMATE_PY_PATH,".env"))
    app.run_sim()