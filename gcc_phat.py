import numpy as np

def gcc_phat(sig_A, sig_B):
    beta = 0
    N = len(sig_A) + len(sig_B) - 1
    Xa = np.fft.rfft(sig_A, n=N)
    Xb = np.fft.rfft(sig_B, n=N)

    numerator = Xa * np.conj(Xb)
    denumerator = np.abs(Xa * np.conj(Xb))
    rms_numerator = np.sqrt(np.mean(np.abs(numerator)**2))
    R = numerator / (denumerator + 1e-10 + beta * rms_numerator)

    cc = np.fft.irfft(R, n=N)
    cc = np.fft.fftshift(cc)
    return cc