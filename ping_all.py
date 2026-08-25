
import os
import shutil

from Floater import *
from bellhop_to_wav import build_ir, read_arr
from coordinate_generator import local_to_geo
import sys
HYDROMATE_PY_PATH = os.getenv('HYDROMATE_PY_PATH')
sys.path.append(HYDROMATE_PY_PATH)
from src.hydromate.app_bellhop import AppBellhop

FS_OUT = 96000
C_SOUND = 1540

def ping_all(Center,f_list):
    FLOATER_NUMBER = len(f_list)
    dist_matrix = np.zeros((FLOATER_NUMBER,FLOATER_NUMBER))

    for i in range(FLOATER_NUMBER):
        for j in range(i):
            TX = f_list[i]
            RX = f_list[j]

            TX_coordinates = local_to_geo(Center,TX.get_gt_position())
            RX_coordinates = local_to_geo(Center,RX.get_gt_position())
            RX_coordinates[[0,1]] = RX_coordinates[[1,0]]
        
            TX_position = np.asarray([TX_coordinates[1],TX_coordinates[0],TX_coordinates[2]])
            app = AppBellhop("ping",
                            np.asarray(TX_position),
                            np.asarray(RX_coordinates),
                            "CVWT", "A",
                            [-89,89],
                            10000,
                            nrd=1)

            app.set_paths("C:/Users/Nicola/Desktop/TESI/HYDROMATE/hm_code_py/.env")
            app.run_sim()

            rr_values, rd_values, arrivals = read_arr('ping_1/ping.arr')
            h, used = build_ir(arrivals, rd_values, max(rr_values), FS_OUT, n_arrivals=0)
            first_non_zero_index = np.nonzero(h)[0][0]

            estimated_distance = (first_non_zero_index / FS_OUT ) * C_SOUND
    
            dist_matrix[i][j] = estimated_distance

            shutil.rmtree(f'ping_1')


    dist_matrix += np.transpose(dist_matrix)
    return dist_matrix

    
def local_ping(Center,position_matrix):
    '''
    print("==POSMATRIX==")
    print(position_matrix)
    print("=============")
    '''
    FLOATER_NUMBER = position_matrix.shape[0]
    dist_matrix = np.zeros((FLOATER_NUMBER,FLOATER_NUMBER))

    for i in range(FLOATER_NUMBER):
        for j in range(i):
            TX_coordinates = local_to_geo(Center,position_matrix[i,:])
            RX_coordinates = local_to_geo(Center,position_matrix[j,:])

            RX_coordinates[[0,1]] = RX_coordinates[[1,0]]
        
            TX_position = np.asarray([TX_coordinates[1],TX_coordinates[0],TX_coordinates[2]])
            app = AppBellhop("ping",
                            np.asarray(TX_position),
                            np.asarray(RX_coordinates),
                            "CVWT", "A",
                            [-89,89],
                            10000,
                            nrd=1)

            app.set_paths("C:/Users/Nicola/Desktop/TESI/HYDROMATE/hm_code_py/.env")
            app.run_sim()

            rr_values, rd_values, arrivals = read_arr('ping_1/ping.arr')
            h, used = build_ir(arrivals, rd_values, max(rr_values), FS_OUT, n_arrivals=0)
            first_non_zero_index = np.nonzero(h)[0][0]

            estimated_distance = (first_non_zero_index / FS_OUT ) * C_SOUND
    
            dist_matrix[i][j] = estimated_distance

            shutil.rmtree(f'ping_1')


    dist_matrix += np.transpose(dist_matrix)
    return dist_matrix


if __name__ == "__main__":
    list = []

    for k in range(5):
        f = Floater(
                    gt_x=0, # Random coordinates
                    gt_y=0,
                    gt_z=0,  # Constant depth
                    dt=1.0,
                    dim=3)
        list.append(f)

    z = ping_all(list)
    print(z)