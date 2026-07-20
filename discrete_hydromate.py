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

def run_discrete_hydromate(Lat_TXs, Lon_TXs, depth_TXs, Lat_RXs, Lon_RXs, depth_RXs, H_index):
    
    import sys
    sys.path.append(r"C:\Users\Nicola\Desktop\TESI\HYDROMATE\hm_code_py")
    from src.hydromate.app_bellhop import AppBellhop

    # Create TMP folder if not exists
    if not os.path.isdir("TMP"):
        os.makedirs("TMP")

    for i,(Lat_TX,Lon_TX,depth_TX,Lat_RX,Lon_RX,depth_RX) in enumerate(zip(Lat_TXs,Lon_TXs,depth_TXs,Lat_RXs, Lon_RXs, depth_RXs)):
        print(f" == TX step number {i+1} ==")

        if NUMBER_OF_HYDROPHONES == 3:
            print()
            #R = eng.NB_prova1Func(Lat_TX,Lon_TX,depth_TX,Lat_RX,Lon_RX,depth_RX)
        elif NUMBER_OF_HYDROPHONES == 4:
            #R = eng.square_NB_runner(Lat_TX,Lon_TX,depth_TX,Lat_RX,Lon_RX,depth_RX)
            print()
        else:
            #R = eng.complete_NB_runner(Lat_TX,Lon_TX,depth_TX,Lat_RX,Lon_RX,depth_RX)


            Floater_hydrophones = get_hydrophones_coordinates(Lat_RX,Lon_RX,depth_RX,5)        
            Floater_hydrophones[:,[0,1]] = Floater_hydrophones[:,[1,0]]
            TX_position = np.asarray([Lon_TX,Lat_TX,depth_TX])

            '''
            print("TX in discrete_hydromate:")
            print(TX_position)
            print("RX in discrete_hydromate:")
            print(Floater_hydrophones)
            '''

            print(np.asarray(TX_position))
            print()
            print(np.asarray(Floater_hydrophones))
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

    #eng.quit()