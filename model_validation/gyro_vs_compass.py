import numpy as np
import matplotlib.pyplot as plt
from digitalshadow.devices.Floater import *

plt.style.use('seaborn-v0_8-deep')

STEPS = 300
FONTSIZE = 14
FONTSIZE_LEGEND = 12

def floater_init():
    f = Floater(1,0,0,0,1,1.0)
    f.Rho_yaw = 1
    f.gt_psi = 0
    f.sigma_yaw_rate = np.deg2rad(0.5)
    #f.sigma_gyro_bias = 0
    return f


if __name__ == "__main__":
    f1 = floater_init()
    f2 = floater_init()
    f1.use_compass = False
    f2.use_compass = True

    psi_gt = []
    psi_gyro = []
    psi_compass = []

    psi_gyro.append(f1.est_psi)
    psi_gt.append(f1.gt_psi)
    psi_compass.append(f2.psi_COMPASS)

    for i in range(STEPS):
        f1.move()
        f2.move()

        if i == 100:
            f1.gt_omega = np.deg2rad(90)
            f2.gt_omega = np.deg2rad(90)
        if i ==101:
            f1.gt_omega = np.deg2rad(0)
            f2.gt_omega = np.deg2rad(0)
        if i == 200:
            f1.gt_omega = np.deg2rad(-90)
            f2.gt_omega = np.deg2rad(-90)
        if i == 201:
            f1.gt_omega = np.deg2rad(0)
            f2.gt_omega = np.deg2rad(0)

        psi_gyro.append(f1.est_psi)
        psi_gt.append(f1.gt_psi)
        psi_compass.append(f2.psi_COMPASS)

    psi_gyro = np.rad2deg(np.asarray(psi_gyro))
    psi_gt = np.rad2deg(np.asarray(psi_gt))
    psi_compass = np.rad2deg(np.asarray(psi_compass))

    fig, ax = plt.subplots(1,3,figsize=(15, 5),sharey=True)

    ax[0].plot(psi_gt,linewidth=2,color="red")
    ax[0].tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax[0].tick_params(axis="both", which="minor", labelsize=FONTSIZE)
    ax[0].set_xlabel("Time [seconds]",fontsize=FONTSIZE)
    ax[0].set_ylabel("Heading [degrees]",fontsize=FONTSIZE)
    ax[0].grid(axis='both',linestyle='--',alpha=0.5)
    ax[0].set_title("Ground truth heading",fontsize=FONTSIZE)

    ax[1].plot(psi_gyro,linewidth=2,color="red")
    ax[1].tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax[1].tick_params(axis="both", which="minor", labelsize=FONTSIZE)
    ax[1].set_xlabel("Time [seconds]",fontsize=FONTSIZE)
    #ax[1].set_ylabel("Heading [degrees]",fontsize=FONTSIZE)
    ax[1].grid(axis='both',linestyle='--',alpha=0.5)
    ax[1].set_title("Gyroscope heading estimation",fontsize=FONTSIZE)

    ax[2].plot(psi_compass,linewidth=2,color="red")
    ax[2].tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax[2].tick_params(axis="both", which="minor", labelsize=FONTSIZE)
    ax[2].set_xlabel("Time [seconds]",fontsize=FONTSIZE)
    #ax[2].set_ylabel("Heading [degrees]",fontsize=FONTSIZE)
    ax[2].grid(axis='both',linestyle='--',alpha=0.5)
    ax[2].set_title("Compass heading estimation",fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig("gyro_vs_compass.png",dpi=900)
    plt.show()