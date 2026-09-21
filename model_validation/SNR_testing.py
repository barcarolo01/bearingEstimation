from digitalshadow.Positioning.coordinate_generator import sposta
from digitalshadow.underwater_sim.discrete_hydromate_single import *
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from utils.utils import clean_temporary_files, compute_single_bearing_angle_complete

if __name__ == "__main__":
    TX_Coordinates = np.asarray([20.832813, 88.698390, 30])
    SAMPLING_FREQUENCY = 96000

    plt.figure()

    distances = np.arange(100,1000,500)
    SNRs = np.arange(-9,1,0.5)

    for SNR in SNRs:
        angle_error = []
        for dist in distances:
            print(f"SNR = {SNR} dB, distace = {dist} m")
            if os.path.isdir("Synth"):
                shutil.rmtree("Synth")
            os.makedirs("Synth")
            
            RX_Coordinates = sposta(TX_Coordinates[:2],dist,90)
            RX_Coordinates = np.asarray([RX_Coordinates[0],RX_Coordinates[1],30])

            run_discrete_hydromate_single(TX_Coordinates[0],
                                            TX_Coordinates[1],
                                            TX_Coordinates[2],
                                            RX_Coordinates[0],
                                            RX_Coordinates[1],
                                            RX_Coordinates[2],
                                            1)

            # Save as wav segment
            for j in range(NUMBER_OF_HYDROPHONES):
                    hydrophone_track = np.load(os.path.join('TMP',f'H{j+1}.npy'))
                    wav.write(f'Synth/T0_F1_H{j+1}.wav', SAMPLING_FREQUENCY, hydrophone_track)

            bearings_err = []
            for k in range(10):
                bearing_arrays, elevation_arrays = compute_single_bearing_angle_complete(timestamp=0, wav_folder='Synth',F_index=1, SNR_desired=SNR, seed = k)
                bearings_err.append(bearing_arrays-180)

            angle_error.append(np.mean(np.asarray(bearings_err)))

        plt.plot(distances,np.abs(angle_error), '-o', markersize=5, label=f"SNR = {SNR} dB")


    plt.legend()
    plt.grid(alpha=0.6)
    plt.savefig("SNR_vs_distance.png")
    plt.xlabel("TX-RX distance [meters]")
    plt.ylabel(r"$| \theta_{real} - \theta_{est} | \;\; [degrees]$")
    plt.show()
    clean_temporary_files()