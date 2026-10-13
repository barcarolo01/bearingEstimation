import numpy as np
import matplotlib.pyplot as plt
from digitalshadow.devices.Floater import *
from digitalshadow.devices.Transmitter import *
from digitalshadow.underwater_sim.ambient_noise import add_ambient_noise, get_noise_config
from digitalshadow.utils.Envelope import detect_env
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

    Center = [32.839, -34.635]
    DISTANCE = 500

    if os.path.isdir('ROCtest'):
        shutil.rmtree('ROCtest')
    os.makedirs('ROCtest')
    os.makedirs(os.path.join('ROCtest','SynthTracks'))

    print(f"Distance = {DISTANCE}")
    TX_Coordinates = local_to_geo(Center,np.asarray([0,0,10]))
    RX_gt_Coordinates = local_to_geo(Center,np.asarray([DISTANCE,0,10]))
    run_discrete_hydromate_single( 'ROCtest',
                                    TX_Coordinates[0],
                                    TX_Coordinates[1],
                                    TX_Coordinates[2],
                                    RX_gt_Coordinates[0],
                                    RX_gt_Coordinates[1],
                                    RX_gt_Coordinates[2],
                                    PSI_RX = 0,
                                    seed = 0)

    for seed in range(N_SIM):
        cfg = get_noise_config()
        # By convolution, obtain a signal for each of the arrival files
        from_arr_to_wav(TX_POS=np.asarray([TX_Coordinates[0],TX_Coordinates[1]]),
                        number_mic=NUMBER_OF_HYDROPHONES,
                        sim_name = 'ROCtest',
                        sim_timestamp = 0,
                        floater_number = 1,
                        n_arrivals=1,
                        sl_db=155,
                        add_noise=False,
                        noise_seed=seed)

        
        rng = np.random.default_rng(seed)

        cleans = []
        fs, clean, = wav.read(os.path.join("ROCtest","SynthTracks","T0_F1_H1.wav"))
        clean = np.zeros(96000)
        cleans.append(clean)
        print(detect_env(clean,fs))
        out, _ = add_ambient_noise(cleans,
                                FS_OUT,shipping=cfg["shipping"],
                                wind=30,
                                extra_db=900,
                                rng=rng,
                                coherent=True)
        out[0] += 100
        wav.write(os.path.join("ROCtest","SynthTracks","T0_F1_H1.wav"),fs,out[0])

        if (clean == out[0]).all():
            print("OMG")

        hydro_signal = out[0]
        print(detect_env(hydro_signal,fs))