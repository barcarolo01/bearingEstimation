from digitalshadow.Positioning.coordinate_generator import sposta
from digitalshadow.underwater_sim.discrete_hydromate_single import *
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from matplotlib import colors
from digitalshadow.utils.utils import *


def wrap_deg(x):
    """Riporta un angolo (o un errore) in [-180, 180)."""
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


if __name__ == "__main__":
    #TX_Coordinates = np.asarray([-21.096, 151.736, 10])
    TX_Coordinates = np.asarray([32.839, -34.635, 30])
    NUMBER_OF_HYDROPHONES = int(os.getenv('NUMBER_OF_HYDROPHONES'))
    SAMPLING_FREQUENCY = 96000

    DISTs = np.array([50, 100, 200, 500, 1000])   # distanze da simulare [m]
    SNRs = np.arange(-18, 4, 1)
    N_TRIALS = 300

    TRUE_BEARING = 180.0
    STAT = "median"          # "median" oppure "rmse": statistica mostrata nel plot

    # errori assoluti: [distanza, SNR, prova]
    all_errors = np.zeros((len(DISTs), len(SNRs), N_TRIALS))

    for d, DIST in enumerate(DISTs):
        print(f"===== Distanza = {DIST} m =====")

        # ---- Simulazione di propagazione: una volta per distanza (non dipende dall'SNR) ----
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
        for i, SNR in enumerate(SNRs):
            print(f"  SNR = {SNR} dB")
            for k in range(N_TRIALS):
                bearing, _ = compute_single_bearing_angle_triangle(
                    timestamp=0, wav_folder='Synth', F_index=1,
                    SNR_desired=SNR,
                    seed=(d * len(SNRs) + i) * N_TRIALS + k)   # seed unico per (distanza, SNR, prova)
                err = wrap_deg(np.mean(bearing) - TRUE_BEARING)
                all_errors[d, i, k] = np.abs(err)

        # salvataggio parziale: se si interrompe, i risultati fin qui non vanno persi
        np.save("abs_errors.npy", all_errors)

    np.save("DISTs.npy", DISTs)
    np.save("SNRs.npy", SNRs)
    np.save("abs_errors.npy", all_errors)       # shape: (len(DISTs), len(SNRs), N_TRIALS)

    # ---- Statistiche (per ogni distanza e SNR) ----
    median = np.median(all_errors, axis=2)                  # (len(DISTs), len(SNRs))
    rmse = np.sqrt(np.mean(all_errors**2, axis=2))
    values = median if STAT == "median" else rmse

    # ---- Plot: una serie per ogni SNR, distanza sull'asse x ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=SNRs.min(), vmax=SNRs.max())

    for i, SNR in enumerate(SNRs):
        ax.plot(DISTs, values[:, i], 'o-', lw=1.5, ms=4,
                color=cmap(norm(SNR)), label=f"{SNR} dB")

    ax.set_xticks(DISTs)                     # un tick per ogni distanza simulata
    ax.set_xticklabels([f"{d:g}" for d in DISTs])
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel(r"$|\theta_{real} - \theta_{est}|$ [deg]")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.6)

    # legenda discreta (una voce per SNR), fuori dall'area del grafico
    ax.legend(title="SNR", loc="center left", bbox_to_anchor=(1.02, 0.5),
              ncol=2, fontsize=8, title_fontsize=9, frameon=True)

    stat_name = "Median" if STAT == "median" else "RMSE"
    ax.set_title(f"Bearing error vs distance ({stat_name})")
    fig.tight_layout()
    fig.savefig(f"bearing_error_vs_distance_{STAT}.png", dpi=500)
    plt.show()
    clean_temporary_files()