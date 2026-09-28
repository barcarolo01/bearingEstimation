from digitalshadow.Positioning.coordinate_generator import sposta
from digitalshadow.underwater_sim.discrete_hydromate_single import *
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from digitalshadow.utils.utils import *


def wrap_deg(x):
    """Riporta un angolo (o un errore) in [-180, 180)."""
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


if __name__ == "__main__":
    TX_Coordinates = np.asarray([20.832813, 88.698390, 30])
    NUMBER_OF_HYDROPHONES = int(os.getenv('NUMBER_OF_HYDROPHONES'))
    SAMPLING_FREQUENCY = 96000

    DIST = 5000
    print(f"DISTANZA = DIST{DIST} metri ")

    if os.path.isdir("Synth"):
        shutil.rmtree("Synth")
    os.makedirs("Synth")

    RX_xy = sposta(TX_Coordinates[:2], DIST, 90)
    RX_Coordinates = np.asarray([RX_xy[0], RX_xy[1], 30])

    run_discrete_hydromate_single(*TX_Coordinates, *RX_Coordinates, 1)

    for j in range(NUMBER_OF_HYDROPHONES):
        track = np.load(os.path.join('TMP', f'H{j+1}.npy'))
        wav.write(f'Synth/T0_F1_H{j+1}.wav', SAMPLING_FREQUENCY, track)