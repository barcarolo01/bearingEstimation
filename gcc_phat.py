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


def gcc_phat_bandlimited(sig_A, sig_B, fs=None, fL=None, fH=None, beta=0):
    N = len(sig_A) + len(sig_B) - 1

    Xa = np.fft.rfft(sig_A, n=N)
    Xb = np.fft.rfft(sig_B, n=N)

    numerator   = Xa * np.conj(Xb)
    denominator = np.abs(numerator)

    # Maschera banda (applicata solo se fs è fornito e almeno uno tra fL/fH è specificato)
    if fs is not None and (fL is not None or fH is not None):
        freqs = np.fft.rfftfreq(N, d=1/fs)
        mask  = np.ones(len(freqs), dtype=bool)
        if fL is not None:
            mask &= freqs >= fL
        if fH is not None:
            mask &= freqs <= fH

        numerator[~mask]   = 0.0
        denominator[~mask] = 0.0

        rms_num = np.sqrt(np.mean(np.abs(numerator)**2))

        R = np.zeros_like(numerator)
        R[mask] = numerator[mask] / (denominator[mask] + 1e-10 + beta * rms_num)
    else:
        rms_num = np.sqrt(np.mean(np.abs(numerator)**2))
        R = numerator / (denominator + 1e-10 + beta * rms_num)

    cc = np.fft.irfft(R, n=N)
    cc = np.fft.fftshift(cc)
    return cc