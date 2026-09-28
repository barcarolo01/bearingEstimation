from digitalshadow.Positioning.coordinate_generator import sposta
from digitalshadow.underwater_sim.discrete_hydromate_single import *
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from digitalshadow.utils.utils import *


def wrap_deg(x):
    """Riporta un angolo (o un errore) in [-180, 180)."""
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


if __name__ == "__main__":
    #TX_Coordinates = np.asarray([-21.096, 151.736, 10])
    TX_Coordinates = np.asarray([32.839, -34.635, 30])
    NUMBER_OF_HYDROPHONES = int(os.getenv('NUMBER_OF_HYDROPHONES'))
    SAMPLING_FREQUENCY = 96000

    DIST = 100
    SNRs = np.arange(-18, 4, 1)
    N_TRIALS = 300

    TRUE_BEARING = 180.0

    # ---- Simulazione di propagazione: UNA sola volta (non dipende dall'SNR) ----
    if os.path.isdir("Synth"):
        shutil.rmtree("Synth")
    os.makedirs("Synth")

    RX_xy = sposta(TX_Coordinates[:2], DIST, 90)
    RX_Coordinates = np.asarray([RX_xy[0], RX_xy[1], 10])

    run_discrete_hydromate_single(*TX_Coordinates, *RX_Coordinates, 1)

    for j in range(NUMBER_OF_HYDROPHONES):
        track = np.load(os.path.join('TMP', f'H{j+1}.npy'))
        wav.write(f'Synth/T0_F1_H{j+1}.wav', SAMPLING_FREQUENCY, track)

    # ---- Monte Carlo sul rumore ----
    all_errors = np.zeros((len(SNRs), N_TRIALS))
    for i, SNR in enumerate(SNRs):
        print(f"SNR = {SNR} dB")
        for k in range(N_TRIALS):
            bearing, _ = compute_single_bearing_angle_triangle(
                timestamp=0, wav_folder='Synth', F_index=1,
                SNR_desired=SNR,
                seed=i * N_TRIALS + k)          # seed diverso per ogni (SNR, prova)
            err = wrap_deg(np.mean(bearing) - TRUE_BEARING)
            all_errors[i, k] = np.abs(err)      # valore assoluto PER PROVA

    np.save("SNRs.npy", SNRs)
    np.save("abs_errors.npy", all_errors)       # salva tutto, non solo la media

    # ---- Statistiche ----
    median = np.median(all_errors, axis=1)
    rmse = np.sqrt(np.mean(all_errors**2, axis=1))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(SNRs, median, 'o-', lw=2, label='Mediana')
    ax.plot(SNRs, rmse, '^--', lw=2, label='RMSE')

    ax.set_xticks(SNRs)                      # un tick per ogni SNR simulato
    ax.set_xticklabels([f"{s:d}" for s in SNRs])
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel(r"$|\theta_{real} - \theta_{est}|$ [deg]")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.6)
    ax.legend()

    ax.set_title(f"Bearing error vs SNR  (distance = {DIST} m)")
    fig.tight_layout()
    fig.savefig("bearing_error_vs_SNR.png", dpi=500)   # dopo aver impostato le etichette
    plt.show()
    clean_temporary_files()