import os
import shutil
from dotenv import load_dotenv
import numpy as np
import matlab.engine
from bellhop_to_wav import build_ir, read_arr

load_dotenv()
NUMBER_OF_HYDROPHONES = int(os.getenv('NUMBER_OF_HYDROPHONES'))
HYDROMATE_PATH = os.getenv('HYDROMATE_PATH')
SAMPLING_FREQUENCY = int(os.getenv('SAMPLING_FREQUENCY'))

def get_floater_distance_ping(Lat_F_1,Lon_F_1,prof_F_1,Lat_F_2,Lon_F_2,prof_F_2):
        """
        Runs a bellhop simulation from one floaters (F1) to another (F2) to simulate a monodirectional
        "ping" message. Assuming a constant speed velocity of 1500 m/s, it estimated the distance
        between the two floaters by evaluating the travel time of the first ray reaching the F2 from F1.
        """
        
        eng = matlab.engine.start_matlab()
        eng.cd(HYDROMATE_PATH, nargout=0)
        eng.NB_ping(Lat_F_1,Lon_F_1,prof_F_1,Lat_F_2,Lon_F_2,prof_F_2)
        shutil.copyfile(f'{HYDROMATE_PATH}/test_ping_1/test_ping.arr',f'TMP/ping.arr')

        rr_vals, rd_vals, arr = read_arr('TMP/ping.arr')
        arrivals = {
                        "rr_vals": rr_vals,
                        "rd_vals": rd_vals,
                        "arr":     arr,
                        "rr_max":  max(rr_vals),
                    }
        
        h, _ = build_ir(arrivals["arr"],
                           arrivals["rd_vals"],
                           arrivals["rr_max"],
                           SAMPLING_FREQUENCY,
                           n_arrivals=0)
  
        c = 1500
        first_arrival_index = np.nonzero(h)[0][0]
        travel_time = first_arrival_index / SAMPLING_FREQUENCY
        distance = c * travel_time
        return distance
        


if __name__ == "__main__":
        estimated_distance = get_floater_distance_ping(34.73,-42.73,20.0,34.73,-42.74,50.0)
        print(estimated_distance)