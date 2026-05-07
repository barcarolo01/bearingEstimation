import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from utils_filters import *
import numpy as np
from utils_filters import *
from utils import *

'''
This method receives as input the initial and final coordinates of the transmitter (vessel)
and an integer n_steps. It returns a list of coordinates equally spaced that connects 
initial coordinates to final ones on a straight line.
'''
def compute_TX_trajectory(Lat_TX_init,Lon_TX_init,Lat_TX_end,Lon_TX_end,n_steps):
    Lat_TXs = np.zeros(n_steps)
    Lon_TXs = np.zeros(n_steps)
    Lat_TXs[0] = Lat_TX_init
    Lon_TXs[0] = Lon_TX_init

    for i in range(1,n_steps):
        Lat_TXs[i] = Lat_TX_init + i*(Lat_TX_end - Lat_TX_init) / (n_steps - 1)
        Lon_TXs[i] = Lon_TX_init + i*(Lon_TX_end - Lon_TX_init) / (n_steps - 1)

    return Lat_TXs, Lon_TXs


'''
This method receives as input the index of a floater.
It analyses the three tracks synthetized for each of the hydrophone of that floater,
applies a window-based analysis and estimates the bearing angle for each window.
An array of bearing angle is produces and saved as a file (numpy array).
'''
def compute_bearing_angle_array(H_index):
    d=0.3
    precompute_bearing_angles(d)
    fs, sig1 = wav.read(f'Synth/H{H_index}_RX1.wav')
    _, sig2 = wav.read(f'Synth/H{H_index}_RX2.wav')
    _, sig3 = wav.read(f'Synth/H{H_index}_RX3.wav')

    # Parametri finestra
    durata_finestra = 0.05 # Secondi
    campioni_finestra = int(durata_finestra * fs)
    print(f"FREQUENZA DI CAMPIONAMENTO: {fs}")
    print(f"FINESTRA: {durata_finestra*1000} ms - {campioni_finestra} samples")
    
    quality_threshold = 0.0
    sample_delay_21, times, tau_percentile_21  = compute_sample_delay_d_aware(sig2,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold)
    sample_delay_32, _, tau_percentile_32 = compute_sample_delay_d_aware(sig3,sig2,fs,campioni_finestra,d,quality_threshold=quality_threshold)
    sample_delay_31, _, tau_percentile_31 = compute_sample_delay_d_aware(sig3,sig1,fs,campioni_finestra,d,quality_threshold=quality_threshold)
    print(f"TAU percentile H2-H1: {tau_percentile_21} samples")
    print(f"TAU percentile H3-H2: {tau_percentile_32} samples")
    print(f"TAU percentile H3-H1: {tau_percentile_31} samples")
    

    datasets_sample_delay = [sample_delay_21,sample_delay_32,sample_delay_31]
    datasets_label = ["tau21","tau32","tau31"]

    fig, axes = plt.subplots(1, 5, figsize=(10, 10), sharey=True)

    time_delay_21 = sample_delay_21 / fs
    time_delay_32 = sample_delay_32 / fs
    time_delay_31 = sample_delay_31 / fs

    # Bearing estimation and error calculation
    tau_fit_error = np.zeros(len(times))
    estimated_bearing = np.zeros(len(times))
    for i in range(len(estimated_bearing)):
        estimated_bearing[i],tau_fit_error[i] = find_bearing(time_delay_32[i],time_delay_21[i],time_delay_31[i])

    np.save(f"Synth/H{H_index}",estimated_bearing)

    return estimated_bearing