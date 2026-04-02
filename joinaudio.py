import numpy as np
from pydub import AudioSegment
import scipy.io.wavfile as wav

def join_audio_files(file1, file2, out_file):
    fs_A, signal_A = wav.read(file1)
    fs_B, signal_B = wav.read(file2)
    
    if(fs_A != fs_B):
        print("Warning: Files have different sampling frequency.")

    conc = np.concatenate((signal_A, signal_B))

    wav.write(out_file,fs_A,conc.astype(np.int16))