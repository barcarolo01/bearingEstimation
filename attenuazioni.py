import os
import shutil
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from bellhop_to_wav import from_arr_to_wav
from coordinate_generator import geo_to_local, local_to_geo, sposta
from discrete_hydromate_single import run_discrete_hydromate_single
from filter_trajectory import *
from findpoint import *
from maps.build_folium_map import build_map
from maps.build_local_map import build_local_cartesian_map
from ping_all import *
from utils_runner import *
from Floater import *

np.random.seed(256123)

SIMULATION_STEPS = 1 
RESURFACE_FREQ = 999999
NUMBER_OF_FLOATERS = 1
TX_LIMIT = 2

SIMULATE = True
SKIPhydromate = False

load_dotenv()
NUMBER_OF_HYDROPHONES = int(os.getenv('NUMBER_OF_HYDROPHONES'))
SAMPLING_FREQUENCY = int(os.getenv('SAMPLING_FREQUENCY'))
HYDROMATE_PY_PATH = os.getenv('HYDROMATE_PY_PATH')

#Center = [20.832813, 88.698390] # India, low depth
#Center = [12.61529, 43.37765]
Center = [39.84164446886851,-70.90061264492535]


Depth_TX = 10
Depth_RX = 10

try:
    os.remove("attenuation_resume.csv")
except FileNotFoundError:
    print(f"Clean.")

with open("attenuation_resume.csv","w") as fw:
    fw.write("distance,mean_amplitude_first_arrival,maximum\n")
    for dist in range(1,700,10):
        amplitudes = []
        strongers= []
        for angle in range(0,360,360):
            rx1 = sposta(Center,dist,angle)

            RX_pos = np.asarray([[[rx1[0],rx1[1],Depth_RX]]])
            Floater_hydrophones = get_hydrophones_coordinates(RX_pos[0,0,0],RX_pos[0,0,1],RX_pos[0,0,2],NUMBER_OF_HYDROPHONES)

            # Swap latitude and longitude (for HYDROMATE compatibility)
            Floater_hydrophones[:,[0,1]] = Floater_hydrophones[:,[1,0]]

            TX_position = np.asarray([Center[1],Center[0],Depth_TX])
            app = AppBellhop("depth_test",
                            np.asarray(TX_position),
                            np.asarray(Floater_hydrophones[0]),
                            "CVWT", "A",
                            [-89,89],
                            10000,
                            nrd=1)
            app.set_paths(os.path.join(HYDROMATE_PY_PATH,".env"))
            app.run_sim()


            # == Reading .arr file
            arr_list  = []   # dictionaries with data from each microphone

            arr_path = os.path.join("depth_test_1", f"depth_test.arr")

            rr_vals, rd_vals, arr = read_arr(arr_path)
            arr_list.append({
                "rr_vals": rr_vals,
                "rd_vals": rd_vals,
                "arr":     arr,
                "rr_max":  max(rr_vals),
            })
            ir_list = []

            h, used = build_ir(arr_list[0]["arr"], arr_list[0]["rd_vals"], arr_list[0]["rr_max"], FS_OUT, n_arrivals=0)
            ir_list.append(h)
            first_not_zero_item = np.nonzero(h)[0][0] # Index of the first non-zero element

            amplitudes.append(h[first_not_zero_item])
            maximo = np.max(h)
            strongers.append(maximo)
            
            if(maximo != h[first_not_zero_item]):
                print("DIFFERENT|")

            print(f"First non zero at {first_not_zero_item}")
            print(f"Strongest value at {np.where(h == maximo)[0][0]}")
            print(f"Amplitude: {h[first_not_zero_item]}")

        mean_amp_per_angle = np.mean(np.asarray(amplitudes))
        max_amp_per_angle = np.mean(np.asarray(maximo))
    
        fw.write(f"{dist},{mean_amp_per_angle},{max_amp_per_angle}\n")