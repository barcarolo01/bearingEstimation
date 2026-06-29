
import numpy as np

c = 1500 # Meters/second

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
    H5 = np.array([0, d/2, -0.236])

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