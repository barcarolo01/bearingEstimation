import io
import os
import shutil
import numpy as np
import scipy.io.wavfile as wav
import matlab.engine
from bellhop_to_wav import from_arr_to_wav
from utils import join_audio_files

def run_discrete_hydromate(Lat_TXs, Lon_TXs, Lat_RX2, Lon_RX2, H_index):
    # Assumption: TX and RXs have the same depth
    depth = 30.0 # Remember the decimal (for MATLAB)

    eng = matlab.engine.start_matlab()
    eng.cd('C:/Users/Nicola/Desktop/TESI/HYDRO2/hm_code', nargout=0)
    if not os.path.isdir("TMP"):
        os.makedirs("TMP")

    for i,(Lat_TX,Lon_TX) in enumerate(zip(Lat_TXs,Lon_TXs)):
        R=eng.NB_prova1Func(Lat_TX,Lon_TX,Lat_RX2,Lon_RX2,depth)
        #eng.NB_launch2hydro(Lat_TX,Lon_TX) 

        # Copy the files from the MATLAB folder to the python folder
        for j in range(3):
            shutil.copyfile(f'C:/Users/Nicola/Desktop/TESI/HYDRO2/hm_code/test_5_{j+1}/test_5.arr',
                            f'C:/Users/Nicola/Desktop/TESI/Prove/TMP/{i}_H{j+1}.arr')

        # By convolution, obtain a .wav file for each of the arrival files
        from_arr_to_wav(rx1=f'TMP/{i}_H1.arr',
                        rx2=f'TMP/{i}_H2.arr',
                        rx3=f'TMP/{i}_H3.arr',
                        source = 'barca.wav',
                        out1 = f'TMP/{i}_H1.wav',
                        out2 = f'TMP/{i}_H2.wav',
                        out3 = f'TMP/{i}_H3.wav',
                        n_arrivals=0)
        
        # Clipping the tracks maintaining only the central portion
        durata = 1 # Seconds
        for j in range(3):
            fs, track = wav.read(f'TMP/{i}_H{j+1}.wav')
            center = len(track) // 2
            #track_clipped = track[center - int(fs/2*(durata)) : center + int(fs/2*(durata))]
            #track_clipped = track[0 : int(96000*(durata))]
            wav.write(f'TMP/{i}_H{j+1}.wav',fs,track.astype(np.int16))
            
        if i == 0:
            for j in range(3):
                shutil.copyfile(f'TMP/{i}_H{j+1}.wav',f'Synth/F{H_index}_H{j+1}.wav')
            
        if i > 0:
            for j in range(3):
                join_audio_files(f'Synth/F{H_index}_H{j+1}.wav',f'TMP/{i}_H{j+1}.wav',f'Synth/F{H_index}_H{j+1}.wav')

    eng.quit()