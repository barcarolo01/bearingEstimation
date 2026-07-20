
from datetime import datetime
import time

import numpy as np

from build_local_map import build_local_cartesian_map
from coordinate_generator import sposta

def local_to_geo(Lat_init,Lon_init,local_x,local_y):
    R = 6371000.0
    Lat = Lat_init + (local_y / R) * (180 / np.pi)
    Lon = Lon_init + (local_x / (R * np.cos(np.radians(Lat_init)))) * (180 / np.pi)
    return (Lat,Lon)
        
def computer_local_mobility(N_STEPS,v_x_init,v_y_init,Rho,sigma_x,sigma_y):
    local_coordinates = np.zeros((N_STEPS,2))
    local_coordinates[0,:] = [0,0]

    v_x = v_x_init
    v_y = v_y_init
    v_x_y = np.sqrt(v_x**2 + v_y**2)

    for i in range(1,N_STEPS):
        e_x = np.random.normal(0,sigma_x)
        e_y = np.random.normal(0,sigma_y)
    
        v_x = v_x*Rho + e_x*np.sqrt(1 - Rho**2)
        v_y = v_y*Rho + e_y*np.sqrt(1 - Rho**2)
        
        local_coordinates[i,:] = [ local_coordinates[i-1,0] + v_x * 1, 
                                   local_coordinates[i-1,1] + v_y * 1 ]


    return np.asarray(local_coordinates)

def computer_RX_mobility(Lat_init,Lon_init,constant_depth,N_STEPS,v_x_init,v_y_init,sigma_x,sigma_y,rho):
    local_coordinates = computer_local_mobility(N_STEPS,v_x_init,v_y_init,rho,sigma_x,sigma_y)
    coordinates_2D = np.asarray(local_to_geo(Lat_init,Lon_init,local_coordinates[:,0],local_coordinates[:,1]))
    depths = constant_depth * np.ones(coordinates_2D.shape[1])
    
    return np.asarray((coordinates_2D[0,:], coordinates_2D[1,:], depths)).T


if __name__ == "__main__":
    np.random.seed(217519)

    Lat_center, Lon_center = 20.832813, 88.698390
    rho = 0.9
    la1,lo1 = sposta(Lat_center,Lon_center,8,225)
    floater_coordinates_1 = computer_RX_mobility(la1,lo1,10,
                                               N_STEPS=100, v_x_init=0, v_y_init=0.5,
                                               sigma_x=0.2,sigma_y=0.2,rho=rho)
    
    la2,lo2 = sposta(Lat_center,Lon_center,7,350)
    floater_coordinates_2 = computer_RX_mobility(la2,lo2,10,
                                               N_STEPS=100, v_x_init=0.5, v_y_init=0,
                                               sigma_x=0.2,sigma_y=0.2,rho=rho)

    la3,lo3 = sposta(Lat_center,Lon_center,10,180)
    floater_coordinates_3 = computer_RX_mobility(la3,lo3,10,
                                               N_STEPS=100, v_x_init=0.25, v_y_init=0.25,
                                               sigma_x=0.2,sigma_y=0.2,rho=rho)

    RX_Coordinates = np.zeros((100,3,3))
    RX_Coordinates[:,0,:] = floater_coordinates_1.copy()
    RX_Coordinates[:,1,:] = floater_coordinates_2.copy()
    RX_Coordinates[:,2,:] = floater_coordinates_3.copy()
    
    build_local_cartesian_map(
        RX_Coordinates,
        None, 
        None, 
        center_coordinates=[Lat_center,Lon_center],
        window_width_m=30, 
        window_height_m=30,
        output_file="map_local.png",
        track_TX=True,
        track_estimated=True
    )
    