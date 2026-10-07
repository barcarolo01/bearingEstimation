import numpy as np
import matplotlib.pyplot as plt

from digitalshadow.devices.Floater import *
from digitalshadow.devices.IMU_models import load_imu_model
from digitalshadow.devices.Transmitter import *
from simulation import *

TX_LIMIT = 3


def angular_error(est_deg, true_deg):
    """Errore angolare assoluto in gradi, gestendo il wrap-around a 360°
    (es. stima 359° con vero 1° -> errore 2°, non 358°)."""
    d = (est_deg - true_deg + 180.0) % 360.0 - 180.0
    return np.abs(d)


if __name__ == '__main__':
    NUMBER_OF_FLOATERS = 2
    STEPS = 1
    N_SIM = 10
    TRUE_BEARING = 180.0
    USE_MEDIAN = True          # True: mediana + banda IQR, False: media + banda ±1 std

    CENTERS = {
        'Red Sea (19.46,38.36)': [19.469223141590106,38.36297468537877],
        'Atlantic ocean (32.84, -34.64)':       [32.839, -34.635],
        'Japan Sea (38.81, 134.06)': [38.81282072005432, 134.05669525507423],
    }

    DISTANCES = np.arange(100, 2000, 50)

    # ERR[c, seed, i] = errore di bearing per centro c, prova seed, distanza i
    ERR = np.full((len(CENTERS), N_SIM, DISTANCES.shape[0]), np.nan)

    for c, (label, Center) in enumerate(CENTERS.items()):
        print(f"=== Centro {c + 1}/{len(CENTERS)}: {label} ===")
        for seed in range(N_SIM):
            print(f"  seed = {seed}")
            for i, L in enumerate(DISTANCES):
                TX = Transmitter(-1, -0, 0, 10, 1.0)
                TX.Rho = 0.999
                TX.set_initial_velocity(20, 0, 0)
                TX.set_sigma(0.0, 0.0, 0)

                floaters = []
                f = Floater(ID=1,
                            gt_x=L,
                            gt_y=0,
                            gt_z=10,
                            NF=NUMBER_OF_FLOATERS,
                            dt=1.0)
                f.use_compass = False
                f.compass_bias = 0
                f.compass_bias_sigma = 0
                floaters.append(f)

                sim = Simulation(
                    Floaters=floaters,
                    Steps=STEPS,
                    Center=Center,
                    transmitter=TX,
                    seed=seed + 99,   # stessi seed per ogni centro -> confronto appaiato
                    sim_name='test')

                try:
                    r = sim.run_simulation()
                    ERR[c, seed, i] = angular_error(r['bearing_arrays'][0][0], TRUE_BEARING)
                except Exception as e:
                    print(f"    [!] errore a L={L}: {e}")   # resta NaN, escluso dalle statistiche

    np.savez('bearing_error_results.npz',
             ERR=ERR, DISTANCES=DISTANCES, centers=list(CENTERS.keys()))

    # ---- Grafico ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for c, label in enumerate(CENTERS.keys()):
        data = ERR[c]
        if USE_MEDIAN:
            center_line = np.nanmedian(data, axis=0)
            lo = np.nanpercentile(data, 25, axis=0)
            hi = np.nanpercentile(data, 75, axis=0)
        else:
            center_line = np.nanmean(data, axis=0)
            sd = np.nanstd(data, axis=0)
            lo, hi = np.maximum(center_line - sd, 0), center_line + sd

        line, = ax.plot(DISTANCES, center_line, marker='o', ms=4, label=label)
        ax.fill_between(DISTANCES, lo, hi, color=line.get_color(), alpha=0.15)

    stat = 'Mediana (banda IQR 25–75%)' if USE_MEDIAN else 'Media (banda ±1σ)'
    ax.set_xlabel('Distanza [m]')
    ax.set_ylabel('Errore di bearing [°]')
    ax.set_title(f'Errore di bearing vs distanza — {stat}, N_SIM={N_SIM}')
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig('bearing_error_vs_distance.png', dpi=150)
    plt.show()