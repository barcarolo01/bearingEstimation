import numpy as np
from floater_geometry import *

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

    E =  (measured_tau21 - lut_tau21) ** 2
    E += (measured_tau32 - lut_tau32) ** 2
    E += (measured_tau31 - lut_tau31) ** 2
    E += (measured_tau41 - lut_tau41) ** 2
    E += (measured_tau42 - lut_tau42) ** 2
    E += (measured_tau43 - lut_tau43) ** 2
    E += (measured_tau51 - lut_tau51) ** 2
    E += (measured_tau52 - lut_tau52) ** 2
    E += (measured_tau53 - lut_tau53) ** 2
    E += (measured_tau54 - lut_tau54) ** 2

    # Find the index of the absolute minimum of the 2D matrix
    flat_min_idx = np.argmin(E)
    az_idx, el_idx = np.unravel_index(flat_min_idx, E.shape)

    # Elevation convertion: from [0 : 180] to [-90 : +90]
    elevation_deg = el_idx - 90  

    return az_idx, elevation_deg