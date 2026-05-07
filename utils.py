import numpy as np
from gcc_phat import *
from utils_filters import *
import numpy as np
import scipy.io.wavfile as wav

c = 1500 # Meters/second

lut_tau32 = np.zeros(360)
lut_tau21 = np.zeros(360)
lut_tau31 = np.zeros(360)
 
def precompute_bearing_angles(d):
    '''
    This function precomputes the time delay expected for each pair of hydrophones
    for each angle between 0° and 359°
    '''
    for angle in range(360):
        theta = angle * np.pi / 180 # Converione gradi --> radianti
        lut_tau32[angle] = -d/c * np.cos(theta)
        lut_tau21[angle] = d/c * np.sin(theta + np.pi/6)
        lut_tau31[angle] = d/c * np.sin(theta - np.pi/6)

def find_bearing(measured_tau32, measured_tau21, measured_tau31):
    """
    This function estimates the angle of arrival (0°-359°) by minimizing the least square error.
    """
    E = ((measured_tau32 - lut_tau32)**2 + (measured_tau21 - lut_tau21)**2 + (measured_tau31 - lut_tau31)**2)
    #E = np.abs(measured_tau32 - lut_tau32) + np.abs(measured_tau21 - lut_tau21) + np.abs(measured_tau31 - lut_tau31)

    estimated_angle = np.argmin(E)

    error = (lut_tau21[estimated_angle]-measured_tau21)**2 + (lut_tau32[estimated_angle]-measured_tau32)**2 + (lut_tau31[estimated_angle]-measured_tau31)**2
    #error = np.abs(lut_tau21[estimated_angle]-measured_tau21) + np.abs(lut_tau32[estimated_angle]-measured_tau32) + np.abs(lut_tau31[estimated_angle]-measured_tau31)

    return estimated_angle,error


def compute_sample_delay(sig_A,sig_B,fs,campioni_finestra,overlap=0.5):
    '''
    This function takes as input two signals and computer the delay (in number of samples)
    by apply a sliding window of a specified length and with a specified degree of overlapping.
    '''
    step = int(campioni_finestra * (1 - overlap))
    n_finestre = 1 + (len(sig_A) - campioni_finestra) // step
    sample_delay = np.zeros(n_finestre)
    times = np.arange(n_finestre) * step / fs

    i = 0
    peaks = []
    for inizio in range(0, min(len(sig_A), len(sig_B)) - int(campioni_finestra), step):
        fine = inizio + campioni_finestra
        finestra1 = sig_A[inizio:fine]
        finestra2 = sig_B[inizio:fine]
        cc = gcc_phat(finestra1, finestra2)
        center = len(cc) // 2
            
        sample_delay[i] = np.argmax(cc) - center  

        i+=1

    return sample_delay,times


def compute_sample_delay_d_aware(sig_A, sig_B, fs, campioni_finestra, d=0.1, c=1500, overlap=0.5, quality_threshold=0.1):
    step = int(campioni_finestra * (1 - overlap))
    n_finestre = 1 + (len(sig_A) - campioni_finestra) // step
    sample_delay = np.zeros(n_finestre)
    times        = np.arange(n_finestre) * step / fs

    # Range fisicamente possibile
    tau_max_samples = int(np.ceil(d / c * fs))+1

    i = 0
    for inizio in range(0, min(len(sig_A), len(sig_B)) - campioni_finestra, step):
        fine      = inizio + campioni_finestra
        finestra1 = sig_A[inizio:fine]
        finestra2 = sig_B[inizio:fine]
        cc     = gcc_phat(finestra1, finestra2)
        center = len(cc) // 2

        # Cerca SOLO nel range fisico ±tau_max_samples
        search = cc[center - tau_max_samples : center + tau_max_samples + 1]
        peak   = np.max(search)


        if peak >= quality_threshold:
            lag = np.argmax(search) - tau_max_samples  # in campioni, relativo a lag=0
        else:
            lag = np.nan   # finestra scartata
        sample_delay[i] = lag
        i += 1

    # 99esimo percentile per stima distanza idrofoni
    tau_percentile = np.nanpercentile(np.abs(sample_delay), 99)

    return sample_delay, times, tau_percentile

def compute_sample_delay_colormap(sig_A, sig_B, fs, campioni_finestra, d=0.1, c=1500, overlap=0.5, quality_threshold=0.1):
    step = int(campioni_finestra * (1 - overlap))
    n_finestre = 1 + (len(sig_A) - campioni_finestra) // step
    sample_delay = np.zeros(n_finestre)
    times        = np.arange(n_finestre) * step / fs

    # Range fisicamente possibile
    tau_max_samples = int(np.ceil(d / c * fs)) + 3
    
    i = 0
    searches = []
    for inizio in range(0, min(len(sig_A), len(sig_B)) - campioni_finestra, step):
        fine      = inizio + campioni_finestra
        finestra1 = sig_A[inizio:fine]
        finestra2 = sig_B[inizio:fine]

        #finestra1 = lowpass_filter_fft(finestra1,fs,30000)
        #finestra2 = lowpass_filter_fft(finestra2,fs,30000)

        cc     = gcc_phat(finestra1, finestra2)
        #cc     = gcc_phat_bandlimited(finestra1, finestra2,fs,10,20000)
        center = len(cc) // 2

        # Cerca SOLO nel range fisico ±tau_max_samples
        search = cc[center - tau_max_samples : center + tau_max_samples + 1]
        peak   = np.max(search)

        if peak >= quality_threshold:
            lag = np.argmax(search) - tau_max_samples  # in campioni, relativo a lag=0
        else:
            lag = np.nan   # finestra scartata
        sample_delay[i] = lag
        searches.append(search)
        i += 1

    # 99esimo percentile per stima distanza idrofoni
    tau_percentile = np.nanpercentile(np.abs(sample_delay), 99)
    searches_np = np.array(searches)
    return searches_np, sample_delay, times, tau_percentile

def add_white_noise(signal, snr_db):
    watt_signal = np.mean(signal**2)
    snr_linear = 10**(snr_db / 10)
    watt_noise = watt_signal / snr_linear
    sigma = np.sqrt(watt_noise)
    noise = np.random.normal(0, sigma, signal.shape)
    return signal + noise


# This method concatenates two .wav files and save the result in out_file
def join_audio_files(file1, file2, out_file):
    fs_A, signal_A = wav.read(file1)
    fs_B, signal_B = wav.read(file2)
    
    if(fs_A != fs_B):
        print("Warning: Files have different sampling frequency.")

    conc = np.concatenate((signal_A, signal_B))

    wav.write(out_file,fs_A,conc.astype(np.int16))