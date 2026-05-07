import os

from Coordinate.move_coordinates import * 
from discrete_hydromate import run_discrete_hydromate
from Coordinate.sposta import sposta
from findpoint import find_points, find_points_3
from mappahtml import build_map
from utils_runner import *

N_steps = 3
N_H = 3

# ==================== Coordinate calculation ====================
# East Australia
Lat_H1_RX2  = -35.9570   
Lon_H1_RX2 = 153.3208 

#Lat_H1_RX2  = 20.832813
#Lon_H1_RX2 = 88.698390 


Lat_H2_RX2, Lon_H2_RX2 = sposta(Lat_H1_RX2,Lon_H1_RX2,2000,"E")
Lat_H3_RX2, Lon_H3_RX2 = sposta(Lat_H2_RX2, Lon_H2_RX2, 2000, "E")

Lat_TX_init, Lon_TX_init = sposta(Lat_H1_RX2, Lon_H1_RX2, 2000, "S")
Lat_TX_end, Lon_TX_end = sposta(Lat_H3_RX2, Lon_H3_RX2, 2000, "S")

print("============ H2 ===============")
print(f"{Lat_H2_RX2}  {Lon_H2_RX2}")
print("===============================")
print("============ TX_init =============")
print(f"{Lat_TX_init}  {Lon_TX_init}")
print("==================================")
print("============ TX_end =============")
print(f"{Lat_TX_end}  {Lon_TX_end}")
print("=================================")


# Obtain the list of intermediate coordinates of the vessel
Lat_TXs, Lon_TXs = compute_TX_trajectory(Lat_TX_init,Lon_TX_init,Lat_TX_end,Lon_TX_end,N_steps)

# ========== Creation of the simulation file ==========
if not os.path.isdir("Synth"):
        os.makedirs("Synth")
with open("Synth/simulation_coordinates.txt", "w") as f:
        f.write(f"{N_H}\n")
        f.write(f"{Lat_H1_RX2} {Lon_H1_RX2}\n")
        f.write(f"{Lat_H2_RX2} {Lon_H2_RX2}\n")
        f.write(f"{Lat_H3_RX2} {Lon_H3_RX2}\n")
        f.write(f"{len(Lat_TXs)}\n")
        for i in range(len(Lat_TXs)):
            if i == len(Lat_TXs) -1:
                 f.write(f"{Lat_TXs[i]} {Lon_TXs[i]}")
            else:
                f.write(f"{Lat_TXs[i]} {Lon_TXs[i]}\n")
# =====================================================

# Hydromate is launched from python: this produces three tracks for each floater of "Synth" folder
run_discrete_hydromate(Lat_TXs, Lon_TXs,Lat_H1_RX2,Lon_H1_RX2,1)
run_discrete_hydromate(Lat_TXs, Lon_TXs,Lat_H2_RX2,Lon_H2_RX2,2)
run_discrete_hydromate(Lat_TXs, Lon_TXs,Lat_H3_RX2,Lon_H3_RX2,3)

# Create the bearing angle array for each floater: this saves
# the bearing angle array of each floater in H{i}.npy.
H1_bearing = compute_bearing_angle_array(1)
H2_bearing = compute_bearing_angle_array(2)
H3_bearing = compute_bearing_angle_array(3)

# Find intersection points: this saves lat-long pairs in "Intersection_coordinates.npy"
'''
intersection_points = find_points((Lat_H1_RX2,Lon_H1_RX2),
                                  (Lat_H2_RX2,Lon_H2_RX2),
                                  H1_bearing,
                                  H2_bearing)
'''

intersection_points = find_points_3((Lat_H1_RX2,Lon_H1_RX2),
                                  (Lat_H2_RX2,Lon_H2_RX2),
                                  (Lat_H3_RX2,Lon_H3_RX2),
                                  H1_bearing,
                                  H2_bearing,
                                  H3_bearing)

np.save("Synth/Intersection_coordinates.npy",intersection_points)