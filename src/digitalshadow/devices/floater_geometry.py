
import math
import numpy as np

c = 1500 # Meters/second

# === Lookup tables ===
lut_tau21 = np.zeros((360, 181))
lut_tau32 = np.zeros((360, 181))
lut_tau31 = np.zeros((360, 181))
lut_tau43 = np.zeros((360, 181))
lut_tau42 = np.zeros((360, 181))
lut_tau41 = np.zeros((360, 181))
lut_tau54 = np.zeros((360, 181))
lut_tau53 = np.zeros((360, 181))
lut_tau52 = np.zeros((360, 181))
lut_tau51 = np.zeros((360, 181))

def precompute_bearing_angles_triangle(d):
    for angle in range(360):
        theta = angle * np.pi / 180 # Converione gradi --> radianti
        lut_tau32[angle,90] = -d/c * np.cos(theta)
        lut_tau21[angle,90] = d/c * np.sin(theta + np.pi/6)
        lut_tau31[angle,90] = d/c * np.sin(theta - np.pi/6)

def precompute_bearing_angles_square(d):
    H1 = [0,d]
    H2 = [d,d]
    H3 = [d,0]
    H4 = [0,0]

    for angle in range(360):
        theta = angle * np.pi / 180 # Converione gradi --> radianti
        lut_tau43[angle,90] = ( (H3[0] - H4[0])*np.cos(theta) + ((H3[1] - H4[1])*np.sin(theta)) ) / c
        lut_tau42[angle,90] = ( (H2[0] - H4[0])*np.cos(theta) + ((H2[1] - H4[1])*np.sin(theta)) ) / c
        lut_tau41[angle,90] = ( (H1[0] - H4[0])*np.cos(theta) + ((H1[1] - H4[1])*np.sin(theta)) ) / c
        lut_tau32[angle,90] = ( (H2[0] - H3[0])*np.cos(theta) + ((H2[1] - H3[1])*np.sin(theta)) ) / c
        lut_tau31[angle,90] = ( (H1[0] - H3[0])*np.cos(theta) + ((H1[1] - H3[1])*np.sin(theta)) ) / c
        lut_tau21[angle,90] = ( (H1[0] - H2[0])*np.cos(theta) + ((H1[1] - H2[1])*np.sin(theta)) ) / c

def precompute_bearing_angles_complete(d):
    H1 = np.array([0,   d,  0])
    H2 = np.array([d,   d,  0])
    H3 = np.array([d,   0,  0])
    H4 = np.array([0,   0,  0])
    H5 = np.array([d/2, 0, -0.236])

    for az_idx in range(360):
        theta = az_idx * np.pi / 180          # azimuth  [0°, 359°]

        for el_idx in range(181):
            phi = (el_idx - 90) * np.pi / 180 # elevazione [-90°, +90°]

            # Versore direzione di propagazione
            dx = np.cos(phi) * np.cos(theta)
            dy = np.cos(phi) * np.sin(theta)
            dz = np.sin(phi)
            direction = np.array([dx, dy, dz])

            # Coppie H1–H4 (z=0, ma manteniamo la formula generale)
            lut_tau21[az_idx, el_idx] = np.dot(H1 - H2, direction) / c
            lut_tau32[az_idx, el_idx] = np.dot(H2 - H3, direction) / c
            lut_tau31[az_idx, el_idx] = np.dot(H1 - H3, direction) / c
            lut_tau41[az_idx, el_idx] = np.dot(H1 - H4, direction) / c
            lut_tau42[az_idx, el_idx] = np.dot(H2 - H4, direction) / c
            lut_tau43[az_idx, el_idx] = np.dot(H3 - H4, direction) / c

            # Coppie con H5 → portano info sull'angolo verticale
            lut_tau51[az_idx, el_idx] = np.dot(H1 - H5, direction) / c
            lut_tau52[az_idx, el_idx] = np.dot(H2 - H5, direction) / c
            lut_tau53[az_idx, el_idx] = np.dot(H3 - H5, direction) / c
            lut_tau54[az_idx, el_idx] = np.dot(H4 - H5, direction) / c



def get_hydrophones_coordinates(lat_center,lon_center,depth_center,number_of_hydrohpones, PSI_RX=0):
    '''
    This method receives as an input the geographical coordinates of the center of a floater and returns 
    the coordinates of each hydrophone
    '''
    # From meters to degrees
    meters_per_deg_lat = 111319.9
    meters_per_deg_lon = 111319.9 * math.cos(math.radians(lat_center))

    psi_rad = math.radians(PSI_RX)
    
    match number_of_hydrohpones:
        case 3:
            L = 0.30 # In meters

            R = L / math.sqrt(3)
            r = L / (2 * math.sqrt(3))

            # H1
            x_H1 = 0.0
            y_H1 = R
            x_H1_rot = x_H1 * math.cos(psi_rad) - y_H1 * math.sin(psi_rad)
            y_H1_rot = x_H1 * math.sin(psi_rad) + y_H1 * math.cos(psi_rad)
            delta_lat_H1 = y_H1_rot / meters_per_deg_lat
            delta_lon_H1 = x_H1_rot / meters_per_deg_lon
            lat_H1 = lat_center + delta_lat_H1
            lon_H1 = lon_center + delta_lon_H1

            # H2
            x_H2 = -(L / 2)
            y_H2 = -r
            x_H2_rot = x_H2 * math.cos(psi_rad) - y_H2 * math.sin(psi_rad)
            y_H2_rot = x_H2 * math.sin(psi_rad) + y_H2 * math.cos(psi_rad)
            delta_lat_H2 = y_H2_rot / meters_per_deg_lat
            delta_lon_H2 = x_H2_rot / meters_per_deg_lon
            lat_H2 = lat_center + delta_lat_H2
            lon_H2 = lon_center + delta_lon_H2

            # H3
            x_H3 = +(L / 2)
            y_H3 = -r
            x_H3_rot = x_H3 * math.cos(psi_rad) - y_H3 * math.sin(psi_rad)
            y_H3_rot = x_H3 * math.sin(psi_rad) + y_H3 * math.cos(psi_rad)
            delta_lat_H3 = y_H3_rot / meters_per_deg_lat
            delta_lon_H3 = x_H3_rot / meters_per_deg_lon
            lat_H3 = lat_center + delta_lat_H3
            lon_H3 = lon_center + delta_lon_H3

            return np.asarray([ (lat_H1, lon_H1,depth_center),
                                (lat_H2, lon_H2,depth_center),
                                (lat_H3, lon_H3,depth_center)])
        
        case 4:
            L = 0.228 / math.sqrt(2) # In meters
            half = L / 2

            # H1: Nord-Ovest
            x_H1 = -half
            y_H1 = half
            x_H1_rot = x_H1 * math.cos(psi_rad) - y_H1 * math.sin(psi_rad)
            y_H1_rot = x_H1 * math.sin(psi_rad) + y_H1 * math.cos(psi_rad)
            delta_lat_H1 = y_H1_rot / meters_per_deg_lat
            delta_lon_H1 = x_H1_rot / meters_per_deg_lon
            lat_H1 = lat_center + delta_lat_H1
            lon_H1 = lon_center + delta_lon_H1

            # H2: Nord-Est
            x_H2 = half
            y_H2 = half
            x_H2_rot = x_H2 * math.cos(psi_rad) - y_H2 * math.sin(psi_rad)
            y_H2_rot = x_H2 * math.sin(psi_rad) + y_H2 * math.cos(psi_rad)
            delta_lat_H2 = y_H2_rot / meters_per_deg_lat
            delta_lon_H2 = x_H2_rot / meters_per_deg_lon
            lat_H2 = lat_center + delta_lat_H2
            lon_H2 = lon_center + delta_lon_H2

            # H3: Sud-Est
            x_H3 = half
            y_H3 = -half
            x_H3_rot = x_H3 * math.cos(psi_rad) - y_H3 * math.sin(psi_rad)
            y_H3_rot = x_H3 * math.sin(psi_rad) + y_H3 * math.cos(psi_rad)
            delta_lat_H3 = y_H3_rot / meters_per_deg_lat
            delta_lon_H3 = x_H3_rot / meters_per_deg_lon
            lat_H3 = lat_center + delta_lat_H3
            lon_H3 = lon_center + delta_lon_H3

            # H4: Sud-Ovest
            x_H4 = -half
            y_H4 = -half
            x_H4_rot = x_H4 * math.cos(psi_rad) - y_H4 * math.sin(psi_rad)
            y_H4_rot = x_H4 * math.sin(psi_rad) + y_H4 * math.cos(psi_rad)
            delta_lat_H4 = y_H4_rot / meters_per_deg_lat
            delta_lon_H4 = x_H4_rot / meters_per_deg_lon
            lat_H4 = lat_center + delta_lat_H4
            lon_H4 = lon_center + delta_lon_H4

            return np.asarray([ (lat_H1, lon_H1,depth_center),
                                (lat_H2, lon_H2,depth_center),
                                (lat_H3, lon_H3,depth_center),
                                (lat_H4, lon_H4,depth_center)])
        
        case 5:
            L = 0.228 / math.sqrt(2) # In meters

            half = L / 2

            # H1: Nord-Ovest
            x_H1 = -half
            y_H1 = half
            x_H1_rot = x_H1 * math.cos(psi_rad) - y_H1 * math.sin(psi_rad)
            y_H1_rot = x_H1 * math.sin(psi_rad) + y_H1 * math.cos(psi_rad)
            delta_lat_H1 = y_H1_rot / meters_per_deg_lat
            delta_lon_H1 = x_H1_rot / meters_per_deg_lon
            lat_H1 = lat_center + delta_lat_H1
            lon_H1 = lon_center + delta_lon_H1

            # H2: Nord-Est
            x_H2 = half
            y_H2 = half
            x_H2_rot = x_H2 * math.cos(psi_rad) - y_H2 * math.sin(psi_rad)
            y_H2_rot = x_H2 * math.sin(psi_rad) + y_H2 * math.cos(psi_rad)
            delta_lat_H2 = y_H2_rot / meters_per_deg_lat
            delta_lon_H2 = x_H2_rot / meters_per_deg_lon
            lat_H2 = lat_center + delta_lat_H2
            lon_H2 = lon_center + delta_lon_H2

            # H3: Sud-Est
            x_H3 = half
            y_H3 = -half
            x_H3_rot = x_H3 * math.cos(psi_rad) - y_H3 * math.sin(psi_rad)
            y_H3_rot = x_H3 * math.sin(psi_rad) + y_H3 * math.cos(psi_rad)
            delta_lat_H3 = y_H3_rot / meters_per_deg_lat
            delta_lon_H3 = x_H3_rot / meters_per_deg_lon
            lat_H3 = lat_center + delta_lat_H3
            lon_H3 = lon_center + delta_lon_H3

            # H4: Sud-Ovest
            x_H4 = -half
            y_H4 = -half
            x_H4_rot = x_H4 * math.cos(psi_rad) - y_H4 * math.sin(psi_rad)
            y_H4_rot = x_H4 * math.sin(psi_rad) + y_H4 * math.cos(psi_rad)
            delta_lat_H4 = y_H4_rot / meters_per_deg_lat
            delta_lon_H4 = x_H4_rot / meters_per_deg_lon
            lat_H4 = lat_center + delta_lat_H4
            lon_H4 = lon_center + delta_lon_H4

            # H5: Middle point of H4-H3
            x_H5 = 0.0
            y_H5 = -half
            x_H5_rot = x_H5 * math.cos(psi_rad) - y_H5 * math.sin(psi_rad)
            y_H5_rot = x_H5 * math.sin(psi_rad) + y_H5 * math.cos(psi_rad)
            delta_lat_H5 = y_H5_rot / meters_per_deg_lat
            delta_lon_H5 = x_H5_rot / meters_per_deg_lon
            lat_H5 = lat_center + delta_lat_H5
            lon_H5 = lon_center + delta_lon_H5

            return np.asarray([ (lat_H1, lon_H1,depth_center),
                                (lat_H2, lon_H2,depth_center),
                                (lat_H3, lon_H3,depth_center),
                                (lat_H4, lon_H4,depth_center),
                                (lat_H5, lon_H5,depth_center+0.236)] )