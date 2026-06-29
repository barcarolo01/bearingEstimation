import os
import shutil
from dotenv import load_dotenv
import numpy as np
import matlab.engine
from bellhop_to_wav import from_arr_to_wav

load_dotenv()
NUMBER_OF_HYDROPHONES = int(os.getenv('NUMBER_OF_HYDROPHONES'))
HYDROMATE_PATH = os.getenv('HYDROMATE_PATH')

def run_discrete_hydromate(Lat_TXs, Lon_TXs, depth_TXs, Lat_RX, Lon_RX, depth_RX, H_index):
    eng = matlab.engine.start_matlab()
    eng.cd(HYDROMATE_PATH, nargout=0)

    # Create TMP folder if not exists
    if not os.path.isdir("TMP"):
        os.makedirs("TMP")

    for i,(Lat_TX,Lon_TX,depth_TX) in enumerate(zip(Lat_TXs,Lon_TXs,depth_TXs)):
        print(f" == TX step number {i+1} ==")

        if NUMBER_OF_HYDROPHONES == 3:
            R = eng.NB_prova1Func(Lat_TX,Lon_TX,depth_TX,Lat_RX,Lon_RX,depth_RX)
        elif NUMBER_OF_HYDROPHONES == 4:
            R = eng.square_NB_runner(Lat_TX,Lon_TX,depth_TX,Lat_RX,Lon_RX,depth_RX)
        else:
            R = eng.complete_NB_runner(Lat_TX,Lon_TX,depth_TX,Lat_RX,Lon_RX,depth_RX)
        

        # Copy the files from the MATLAB folder to the python folder
        for j in range(NUMBER_OF_HYDROPHONES):
            shutil.copyfile(f'{HYDROMATE_PATH}/test_5_{j+1}/test_5.arr',
                            f'C:/Users/Nicola/Desktop/TESI/Prove/TMP/H{j+1}.arr')

        # By convolution, obtain a signal for each of the arrival files
        from_arr_to_wav(input_folder='TMP',
                        number_mic=NUMBER_OF_HYDROPHONES,
                        source = 'AudioFiles/barca.wav',
                        out_folder = 'TMP',
                        n_arrivals=0)
                 
        if i == 0:
            for j in range(NUMBER_OF_HYDROPHONES):
                shutil.copyfile(f'TMP/H{j+1}.npy',f'Synth/F{H_index}_H{j+1}.npy')
            
        if i > 0:
            for j in range(NUMBER_OF_HYDROPHONES):
                audio_array = np.load(f'Synth/F{H_index}_H{j+1}.npy')
                last_step = np.load(f'TMP/H{j+1}.npy')
                joined_array = np.concatenate([audio_array,last_step])
                np.save(f'Synth/F{H_index}_H{j+1}.npy',joined_array)

    eng.quit()