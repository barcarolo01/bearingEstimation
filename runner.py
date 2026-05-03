from Coordinate.move_coordinates import * 
from discrete_hydromate import run_discrete_hydromate
from Coordinate.sposta import sposta
from findpoint import find_points
from mappahtml import build_map
from utils_runner import *

N_steps = 11

# ==================== Coordinate calculation ====================
Lat_H1_RX2  = -35.9570   
Lon_H1_RX2 = 153.3208  

'''
Lat_H2_RX2, Lon_H2_RX2 = sposta_coordinate_vincenty(Lat_H1_RX2, Lon_H1_RX2, 200, direzione_a_bearing("S"))
lat_tmp, lon_tmp = sposta_coordinate_vincenty(Lat_H1_RX2, Lon_H1_RX2, 100, direzione_a_bearing("S"))
Lat_TX_init, Lon_TX_init = sposta_coordinate_vincenty(lat_tmp, lon_tmp, 100, direzione_a_bearing("E"))
Lat_TX_end, Lon_TX_end = sposta_coordinate_vincenty(Lat_TX_init, Lon_TX_init, 200, direzione_a_bearing("O"))
'''

Lat_H2_RX2, Lon_H2_RX2 = sposta(Lat_H1_RX2, Lon_H1_RX2, 1000, "S")
#lat_tmp, lon_tmp = sposta(Lat_H1_RX2, Lon_H1_RX2, 100, "S")
Lat_TX_init, Lon_TX_init = sposta(Lat_H1_RX2, Lon_H1_RX2, 1000, "E")
Lat_TX_end, Lon_TX_end = sposta(Lat_H2_RX2, Lon_H2_RX2, 1000, "O")

print("============ H2 ===============")
print(f"{Lat_H2_RX2}  {Lon_H2_RX2 }")
print("===============================")
print("============ TX_init =============")
print(f"{Lat_TX_init}  {Lon_TX_init }")
print("==================================")
print("============ TX_end =============")
print(f"{Lat_TX_end}  {Lon_TX_end }")
print("=================================")

Lat_TXs, Lon_TXs = compute_TX_trajectory(Lat_TX_init,Lon_TX_init,Lat_TX_end,Lon_TX_end,N_steps)

run_discrete_hydromate(Lat_TXs, Lon_TXs,Lat_H1_RX2,Lon_H1_RX2,1)
run_discrete_hydromate(Lat_TXs, Lon_TXs,Lat_H2_RX2,Lon_H2_RX2,2)

H1_bearing = compute_bearing_angle_array(1)
H2_bearing = compute_bearing_angle_array(2)

intersection_points = find_points((Lat_H1_RX2,Lon_H1_RX2),
                                  (Lat_H2_RX2,Lon_H2_RX2),
                                  H1_bearing,
                                  H2_bearing)

Lat_intersected = intersection_points[:,0]
Lon_intersected = intersection_points[:,1]

build_map((Lat_H1_RX2,Lon_H1_RX2),
          (Lat_H2_RX2,Lon_H2_RX2),
          10,
          10,
          Lat_TXs,
          Lon_TXs,
          Lat_intersected,
          Lon_intersected)