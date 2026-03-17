from gcc_phat import *
from scipy.signal import butter, sosfilt

def lowpass_filter(signal, fc, fs, order=4):
    sos = butter(order, fc, btype='low', fs=fs, output='sos')
    return sosfilt(sos, signal)

def highpass_filter(signal, fc, fs, order=4):
    sos = butter(order, fc, btype='high', fs=fs, output='sos')
    return sosfilt(sos, signal)

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sosfilt(sos, signal)