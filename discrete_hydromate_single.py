import os
import shutil
from dotenv import load_dotenv
import hydromate
import numpy as np
from bellhop_to_wav import from_arr_to_wav
#from hydromate.app_bellhop import AppBellhop
from floater_geometry import get_hydrophones_coordinates

load_dotenv()
NUMBER_OF_HYDROPHONES = int(os.getenv('NUMBER_OF_HYDROPHONES'))
HYDROMATE_PATH = os.getenv('HYDROMATE_PATH')

def run_discrete_hydromate_single(Lat_TX, Lon_TX, depth_TX, Lat_RX, Lon_RX, depth_RX, H_index):
    
    import sys
    sys.path.append(r"C:\Users\Nicola\Desktop\TESI\HYDROMATE\hm_code_py")
    from src.hydromate.app_bellhop import AppBellhop

    # Create TMP folder if not exists
    if not os.path.isdir("TMP"):
        os.makedirs("TMP")

    # NUMBER_OF_HYDROPHONES check
    if NUMBER_OF_HYDROPHONES not in [3,4,5]:
        raise ValueError(f"NUMBER_OF_HYDROPHONES bust be either 3, 4 or 5 (while it is {NUMBER_OF_HYDROPHONES}).")

    Floater_hydrophones = get_hydrophones_coordinates(Lat_RX,Lon_RX,depth_RX,NUMBER_OF_HYDROPHONES)
    
    # Swap latitude and longitude (for HYDROMATE compatibility)
    Floater_hydrophones[:,[0,1]] = Floater_hydrophones[:,[1,0]]

    TX_position = np.asarray([Lon_TX,Lat_TX,depth_TX])
    app = AppBellhop("HM_out",
                    np.asarray(TX_position),
                    np.asarray(Floater_hydrophones),
                    "CVWT", "A",
                    [-89,89],
                    10000,
                    nrd=1)

    app.set_paths("C:/Users/Nicola/Desktop/TESI/HYDROMATE/hm_code_py/.env")
    app.run_sim()

    

    # Copy the files from the MATLAB folder to the python folder
    for j in range(NUMBER_OF_HYDROPHONES):
        shutil.copyfile(f'HM_out_{j+1}/HM_out.arr',
                        f'C:/Users/Nicola/Desktop/TESI/Prove/TMP/H{j+1}.arr')
        shutil.rmtree(f'HM_out_{j+1}')

    # By convolution, obtain a signal for each of the arrival files
    from_arr_to_wav(input_folder='TMP',
                    number_mic=NUMBER_OF_HYDROPHONES,
                    source = 'AudioFiles/barca.wav',
                    out_folder = 'TMP',
                    n_arrivals=0)


    # If the file does not exist, it means it is the first simulation step for this floater
    if not os.path.isfile(f'Synth/F{H_index}_H1.npy'): 
        for j in range(NUMBER_OF_HYDROPHONES):
            shutil.copyfile(f'TMP/H{j+1}.npy',f'Synth/F{H_index}_H{j+1}.npy')
    else:
        for j in range(NUMBER_OF_HYDROPHONES):
            audio_array = np.load(f'Synth/F{H_index}_H{j+1}.npy')
            last_step = np.load(f'TMP/H{j+1}.npy')
            joined_array = np.concatenate([audio_array,last_step])
            np.save(f'Synth/F{H_index}_H{j+1}.npy',joined_array)