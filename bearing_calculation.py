import numpy as np
from precompute_LUTs import *

def find_bearing_triangle(measured_tau32, measured_tau21, measured_tau31):
    """
    This function estimates the angle of arrival (0°-359°) by minimizing the least square error.
    """
    E  = (measured_tau32 - lut_tau32[:,90])**2
    E += (measured_tau21 - lut_tau21[:,90])**2
    E += (measured_tau31 - lut_tau31[:,90])**2
    estimated_angle = np.argmin(E)
    error = (lut_tau21[estimated_angle,90]-measured_tau21)**2 + (lut_tau32[estimated_angle,90]-measured_tau32)**2 + (lut_tau31[estimated_angle,90]-measured_tau31)**2

    return estimated_angle,error

def find_bearing_square(measured_tau32, measured_tau21, measured_tau31,measured_tau41,measured_tau42,measured_tau43):
    E  = (measured_tau32 - lut_tau32[:,90])**2
    E += (measured_tau21 - lut_tau21[:,90])**2 
    E += (measured_tau31 - lut_tau31[:,90])**2 
    E += (measured_tau41 - lut_tau41[:,90])**2
    E += (measured_tau42 - lut_tau42[:,90])**2
    E += (measured_tau43 - lut_tau43[:,90])**2
    estimated_angle = np.argmin(E)

    return estimated_angle


def find_bearing_complete(measured_tau32, measured_tau21, measured_tau31,
                          measured_tau41, measured_tau42, measured_tau43,
                          measured_tau51, measured_tau52, measured_tau53, measured_tau54):
    """
    Restituisce (azimuth_deg, elevation_deg).
    - azimuth   in [0°, 359°]  → stimato da find_bearing_square
    - elevation in [-90°, +90°] → ricerca 1D lungo la colonna dell'azimuth stimato
    """
    # Azimuth is computed using only the information of the planar hydrophones
    az_idx = find_bearing_square(measured_tau32, measured_tau21, measured_tau31,
                                 measured_tau41, measured_tau42, measured_tau43)

    # Estimates the depth starting from the aximuth computed before
    E_el  = (measured_tau51 - lut_tau51[az_idx, :]) ** 2
    E_el += (measured_tau52 - lut_tau52[az_idx, :]) ** 2
    E_el += (measured_tau53 - lut_tau53[az_idx, :]) ** 2
    E_el += (measured_tau54 - lut_tau54[az_idx, :]) ** 2

    el_idx = np.argmin(E_el)

    # Converting [0 : 180] to [-90 : +90]
    elevation_deg = el_idx - 90  
    return az_idx, elevation_deg