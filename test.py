import numpy as np
import matplotlib.pyplot as plt
from digitalshadow.devices.Floater import *
from digitalshadow.devices.Transmitter import *
from simulation import *

TX_LIMIT = 3

def angular_error(est_deg, true_deg):
    """Errore angolare assoluto in gradi, gestendo il wrap-around a 360°
    (es. stima 359° con vero 1° -> errore 2°, non 358°)."""
    d = (est_deg - true_deg + 180.0) % 360.0 - 180.0
    return np.abs(d)


if __name__ == '__main__':
    N_SIM = 5
    TRUE_BEARING = 180.0
    USE_MEDIAN = True          # True: mediana + banda IQR, False: media + banda ±1 std

    CENTERS = {
        'MEX': [28.16588430899801,-95.30713414731443],
        'India': [20.832813, 88.698390],
        'Atlantic ocean': [32.839, -34.635],

        #'Japan Sea (38.81, 134.06)': [38.81282072005432, 134.05669525507423],
    }

    DISTANCES = np.arange(50, 2000, 50)
    #DISTANCES[-1] = 10000
    
    # ERR[c, seed, i] = errore di bearing per centro c, prova seed, distanza i
    ERR = np.full((len(CENTERS), N_SIM, DISTANCES.shape[0]), np.nan)

    if os.path.isdir('test_distance'):
        shutil.rmtree('test_distance')
    os.makedirs('test_distance')
    os.makedirs(os.path.join('test_distance','SynthTracks'))

    for c, (label, Center) in enumerate(CENTERS.items()):
        print(f"=== Centro {c + 1}/{len(CENTERS)}: {label} ===")
        for i, L in enumerate(DISTANCES):
            print(f"Distance = {L}")
            TX_Coordinates = local_to_geo(Center,np.asarray([0,0,10]))
            RX_gt_Coordinates = local_to_geo(Center,np.asarray([L,0,10]))
            run_discrete_hydromate_single( 'test_distance',
                                            TX_Coordinates[0],
                                            TX_Coordinates[1],
                                            TX_Coordinates[2],
                                            RX_gt_Coordinates[0],
                                            RX_gt_Coordinates[1],
                                            RX_gt_Coordinates[2],
                                            PSI_RX = 0,
                                            seed = 0)

            for seed in range(N_SIM):
                # By convolution, obtain a signal for each of the arrival files
                from_arr_to_wav(TX_POS=np.asarray([TX_Coordinates[0],TX_Coordinates[1]]),
                                number_mic=NUMBER_OF_HYDROPHONES,
                                sim_name = 'test_distance',
                                sim_timestamp = 0,
                                floater_number = 1,
                                n_arrivals=1,
                                sl_db=155,
                                add_noise=False,
                                noise_seed=seed)
                
                bearing_arrays, elevation_arrays  = compute_single_bearing_angle_triangle(track_folder=os.path.join('test_distance','SynthTracks'), timestamp=0, F_index=1)
                print(bearing_arrays)
                ERR[c, seed, i] = angular_error(bearing_arrays, TRUE_BEARING)
           
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
    fig.savefig('bearing_error_vs_distance.png', dpi=300)
    plt.show()