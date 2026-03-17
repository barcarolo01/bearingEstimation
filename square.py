import numpy as np

lut_tau41 = np.zeros(360)
lut_tau42 = np.zeros(360)
lut_tau43 = np.zeros(360)
lut_tau32 = np.zeros(360)
lut_tau31 = np.zeros(360)
lut_tau21 = np.zeros(360)

c = 1500

def precompute_bearing_angles(d):
    H1 = [0,d]
    H2 = [d,d]
    H3 = [d,0]
    H4 = [0,0]

    for angle in range(360):
        theta = angle * np.pi / 180 # Converione gradi --> radianti
        lut_tau43[angle] = ( (H4[0] - H3[0])*np.cos(theta) + ((H4[1] - H3[1])*np.sin(theta)) ) / c
        lut_tau42[angle] = ( (H4[0] - H2[0])*np.cos(theta) + ((H4[1] - H2[1])*np.sin(theta)) ) / c
        lut_tau41[angle] = ( (H4[0] - H1[0])*np.cos(theta) + ((H4[1] - H1[1])*np.sin(theta)) ) / c
        lut_tau32[angle] = ( (H3[0] - H2[0])*np.cos(theta) + ((H3[1] - H2[1])*np.sin(theta)) ) / c
        lut_tau31[angle] = ( (H3[0] - H1[0])*np.cos(theta) + ((H3[1] - H1[1])*np.sin(theta)) ) / c
        lut_tau21[angle] = ( (H2[0] - H1[0])*np.cos(theta) + ((H2[1] - H1[1])*np.sin(theta)) ) / c

    print(lut_tau43[0])

if __name__ == "__main__":
    precompute_bearing_angles(1)