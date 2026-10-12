
from matplotlib.pylab import rfft, rfftfreq
import numpy as np
import soundfile as sf

def band_fraction_db(sig, fs, f_low, f_high, f_bb_low=None, f_bb_high=None):
    """
    Fraction of the track's power that falls inside [f_low, f_high], in dB.
    f_bb_low/f_bb_high delimit the band the published broadband SL refers to
    (e.g. 23 Hz - 24 kHz for HearMyShip); default is the whole Nyquist band.
    """
    S = rfft(sig)
    f = rfftfreq(len(sig), 1 / fs)
    ref = np.ones(len(f), bool)
    if f_bb_low  is not None: ref &= f >= f_bb_low
    if f_bb_high is not None: ref &= f <= f_bb_high
    sel = ref & (f >= f_low) & (f <= f_high)
    return 10 * np.log10(np.sum(np.abs(S[sel]) ** 2) / np.sum(np.abs(S[ref]) ** 2))

 
def sl_band_from_broadband(sig, fs, sl_bb_db, f_low, f_high,
                           f_bb_low=23.0, f_bb_high=24000.0, verbose=True):
    """
    In-band source level [dB re 1 uPa @ 1 m] from a published BROADBAND source
    level, using the spectral shape of the source track itself:
 
        SL_band = SL_bb + 10 log10( P[f_low..f_high] / P[f_bb_low..f_bb_high] )
 
    Exact when the track is a real recording of the same kind of vessel, since
    the recording already carries the right spectral shape. Pass the result as
    sl_db to load_audio_source(): the track is band-limited there, so its RMS
    then equals the in-band level.
    """
    frac = band_fraction_db(sig, fs, f_low, f_high, f_bb_low, f_bb_high)
    sl_band = sl_bb_db + frac
    if verbose:
        print(f"SL broadband      = {sl_bb_db:6.1f} dB re 1 uPa @ 1 m "
              f"({f_bb_low:.0f} Hz - {f_bb_high/1000:.0f} kHz)")
        print(f"frazione in banda = {frac:+6.1f} dB "
              f"({f_low/1000:.0f}-{f_high/1000:.0f} kHz)")
        print(f"SL in banda       = {sl_band:6.1f} dB re 1 uPa @ 1 m")
    return sl_band

if __name__ == '__main__':
    raw, fs_raw = sf.read("AudioFiles/motorboat.wav")          # il wav ORIGINALE, non normalizzato in banda
    S = np.abs(rfft(raw))**2
    f = rfftfreq(len(raw), 1/fs_raw)
    tot = S.sum()
    for lo, hi in [(20,250),(250,1000),(1000,2500),(2500,5000),
                (5000,10000),(10000,18000),(18000,48000)]:
        m = (f >= lo) & (f < hi)

        print(f"{lo:6d}-{hi:6d} Hz: {10*np.log10(S[m].sum()/tot):+6.1f} dB")
    
    sl_band = sl_band_from_broadband(raw, fs_raw, sl_bb_db=150.0,
                                    f_low=5000, f_high=18000)