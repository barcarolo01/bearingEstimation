from digitalshadow.devices.Floater import *
from digitalshadow.devices.Transmitter import *
from simulation import *
import matplotlib.pyplot as plt

FONTSIZE = 14
FONTSIZE_LEGEND = 12

DIST = 100
SNRs = np.arange(-15, 3, 1)
N_TRIALS = 200

SIM_NAME = "SNR_testing"

def wrap_deg(x):
    """Riporta un angolo (o un errore) in [-180, 180)."""
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


if __name__ == "__main__":
    Center = [29.834027625426177, 49.69442749625741]
    
    print(f"Depth = {get_depth_gebco(Center[0],Center[1])}")

    TX = Transmitter(ID=-1,gt_x=0,gt_y=0,gt_z=3,dt=1.0)
    floaters = []
    RX = Floater(ID=1,gt_x=100,gt_y=0,gt_z=3,NF=1,dt=1)
    floaters.append(RX)


    #TX_Coordinates = np.asarray([38.81282072005432,134.05669525507423, 30])
    NUMBER_OF_HYDROPHONES = int(os.getenv('NUMBER_OF_HYDROPHONES'))
    SAMPLING_FREQUENCY = 96000


    TRUE_BEARING = 180.0

    sim = Simulation(transmitter=TX,
                     Floaters=floaters,
                     Steps=1,
                     Center=Center,
                     sim_name=SIM_NAME)
    
    sim.run_simulation()

    # ---- Monte Carlo sul rumore ----
    all_errors = np.zeros((len(SNRs), N_TRIALS))
    for i, SNR in enumerate(SNRs):
        print(f"SNR = {SNR} dB")
        for k in range(N_TRIALS):
            bearing, _ = compute_single_bearing_angle_triangle(
                timestamp=0, track_folder=os.path.join(SIM_NAME,'SynthTracks'),
                F_index=1,
                SNR_desired=SNR,
                seed=i * N_TRIALS + k +1)  
            err = np.mean(bearing) - TRUE_BEARING
            all_errors[i, k] = np.abs(err)      # valore assoluto PER PROVA

    np.save("SNRs.npy", SNRs)
    np.save("abs_errors.npy", all_errors)       # salva tutto, non solo la media

    # ---- Statistiche ---- 
    median = np.median(all_errors, axis=1)
    q25 = np.percentile(all_errors, 25, axis=1)
    q75 = np.percentile(all_errors, 75, axis=1)
    fig, ax = plt.subplots(figsize=(12, 5))
    

    ax.plot(SNRs, median, 'o-', lw=2, label='Mediana')
    ax.fill_between(SNRs, q25, q75, alpha=0.3, label='IQR')
    ax.legend()
    ax.set_xticks(SNRs)             
    ax.set_xticklabels([f"{s:d}" for s in SNRs])
    ax.set_xlabel("SNR [dB]",fontsize=FONTSIZE)
    ax.set_ylabel(r"$|\theta_{real} - \theta_{est}|$ [deg]",fontsize=FONTSIZE)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE_LEGEND)

    ax.set_title(f"Bearing error vs SNR  (distance = {DIST} m)")
    fig.tight_layout()
    fig.savefig("bearing_error_vs_SNR.png", dpi=900)
    plt.show()
    clean_temporary_files(SIM_NAME)