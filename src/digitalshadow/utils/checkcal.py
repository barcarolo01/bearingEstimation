import numpy as np

P_REF = 1e-6  # Pa (1 µPa)

AMP_IDX = 0


def _active_rms(x, fs, win_s=0.05, thr_db=-20.0):
    """RMS calcolato solo sulle finestre in cui c'e' segnale
    (energia entro thr_db dal massimo), per escludere silenzi e code a zero."""
    x = np.asarray(x, dtype=float)
    n = max(1, int(win_s * fs))
    n_win = len(x) // n
    if n_win == 0:
        return np.sqrt(np.mean(x ** 2))
    p = np.mean(x[:n_win * n].reshape(n_win, n) ** 2, axis=1)
    active = p > p.max() * 10 ** (thr_db / 10)
    return np.sqrt(np.mean(p[active]))


def check_calibration(rx, src, used, fs, sl_db, label=""):
    """
    Verifica che la traccia ricevuta rx [Pa] sia coerente con
        RL = SL - TL
    dove TL e' la transmission loss incoerente calcolata dagli arrivi di Bellhop:
        TL_inc = -10 log10( sum |A_k|^2 )

    rx    : traccia simulata [Pa]
    src   : segnale sorgente gia' scalato da load_audio_source [Pa @ 1 m]
    used  : arrivi selezionati per questo idrofono
    fs    : frequenza di campionamento
    sl_db : SL richiesto, dB re 1 µPa @ 1 m
    """
    amps = np.array([abs(a[AMP_IDX]) for a in used], dtype=float)
    tl_inc = -10 * np.log10(np.sum(amps ** 2))

    sl_meas = 20 * np.log10(_active_rms(src, fs) / P_REF)   # SL effettivo della sorgente
    rl_meas = 20 * np.log10(_active_rms(rx, fs) / P_REF)    # livello ricevuto
    rl_exp = sl_meas - tl_inc
    diff = rl_meas - rl_exp

    sl_str = f"{sl_db:6.1f}" if sl_db is not None else "  None"
    print(f"[{label:>4}] SL target {sl_str} | SL measured {sl_meas:6.1f} | "
          f"TL {tl_inc:6.1f} | RL targer {rl_exp:6.1f} | "
          f"RL measured {rl_meas:6.1f} dB re 1 µPa | diff {diff:+5.1f} dB")
    return diff