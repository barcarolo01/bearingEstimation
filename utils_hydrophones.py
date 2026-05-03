import numpy as np
from gcc_phat import *
from utils_filters import *
from utils import gcc_phat

c = 1500 # Meters/second
tau_thr = 0.001

lut_tau23 = np.zeros(360)
lut_tau21 = np.zeros(360)
lut_tau31 = np.zeros(360)

def precompute_bearing_angles(d):
    '''
    This function precomputes the time delay expected for each pair of hydrophones
    for each angle between 0° and 359°
    '''
    for angle in range(360):
        theta = angle * np.pi / 180 # Converione gradi --> radianti
        lut_tau23[angle] = d/c * np.cos(theta)
        lut_tau21[angle] = d/c * np.sin(theta + np.pi/6)
        lut_tau31[angle] = d/c * np.sin(theta - np.pi/6)

def find_bearing(measured_tau23, measured_tau21, measured_tau31):
    """
    This function estimates the angle of arrival (0°-259°) by minimizing the least square error.
    """
    E = ((measured_tau23 - lut_tau23)**2 + (measured_tau21 - lut_tau21)**2 + (measured_tau31 - lut_tau31)**2)
    to_be_returned = np.argmin(E)
    least_square_error = np.abs(lut_tau21[to_be_returned]-measured_tau21) + np.abs(lut_tau23[to_be_returned]-measured_tau23) + np.abs(lut_tau31[to_be_returned]-measured_tau31)

    return to_be_returned,least_square_error

def compute_sample_delay(sig_A,sig_B,fs,campioni_finestra,overlap=0):
    '''
    This function takes as input two signals and computer the delay (in number of samples)
    by apply a sliding window of a specified length and with a specified degree of overlapping.
    '''
    step = int(campioni_finestra * (1 - overlap))
    n_finestre = 1 + (len(sig_A) - campioni_finestra) // step
    sample_delay = np.zeros(n_finestre)
    times = np.arange(n_finestre) * step / fs

    starts = np.arange(n_finestre) * step
    i = 0
    for i, inizio in enumerate(starts):
        fine = inizio + campioni_finestra
        finestra1 = sig_A[inizio:fine]
        finestra2 = sig_B[inizio:fine]
        cc = gcc_phat(finestra1, finestra2)

        center = len(cc) // 2
        sample_delay[i] = np.argmax(cc) - center     
        
        i+=1
    return sample_delay,times