import numpy as np
import matplotlib.pyplot as plt
from digitalshadow.devices.Floater import *

plt.style.use('seaborn-v0_8-deep')

NUMBER_OF_RUNS = 20
STEPS = 900

FONTSIZE = 14
FONTSIZE_LEGEND = 12

ALPHAS = [0.0, 0.5, 0.9, 0.98, 0.995, 0.999]

def run_single_simulation(f, steps):
    err = np.zeros(steps)
    gt_pos = np.zeros((steps,3))
    est_pos = np.zeros((steps,3))
    for k in range(steps):
        f.move()
        err[k] = np.linalg.norm(f.gt_pos[:2] - f.est_pos[:2])
        gt_pos[k,:] = f.gt_pos
        est_pos[k,:] = f.est_pos

    return np.sqrt(np.mean(err**2, axis=0)), err, gt_pos, est_pos


if __name__ == "__main__":
    """Still floater; alpha = None means gyroscope only (no compass)."""
    rms_values = np.zeros(NUMBER_OF_RUNS)
    err_arrays = np.zeros((NUMBER_OF_RUNS,STEPS))
    gt_arrays = np.zeros((NUMBER_OF_RUNS,STEPS,3))
    est_arrays = np.zeros((NUMBER_OF_RUNS,STEPS,3))

    for i in range(NUMBER_OF_RUNS):
        #rnd_run = np.random.default_rng(10+i**2)

        f = Floater(0, 0, 0, 0, 1, 1.0, imu_seed=i)
        f.set_initial_velocity(0.5, 0.5, 0.0)
        f.Rho = 0.999
        f.Rho_yaw = 0.999
        f.set_sigma(0.1, 0.1, 0.0)
        f.sigma_yaw_rate = 0.0
        f.use_compass = False

        f.accel_bias = np.zeros(3)
        f.sigma_accel_white_noise = np.zeros(3)
        f.sigma_accel_bias_driving = np.zeros(3)

        f.gyro_bias = 0
        f.sigma_gyro_white_noise = 0
        f.sigma_gyro_bias_driving = 0

        t = np.arange(1, STEPS + 1)
        rms_values[i], err_arrays[i,:], gt_arrays[i,:,:], est_arrays[i,:,:] = run_single_simulation(f,STEPS)

        if i < 1:
            plt.figure()
            plt.plot(gt_arrays[i,:,0],gt_arrays[i,:,1], linewidth=2, color = 'green', label='Grount truth')
            plt.plot(est_arrays[i,:,0],est_arrays[i,:,1], linewidth=2, color = 'red', label='Estimated') 
            plt.xlabel("Easting [meters]",fontsize=FONTSIZE)
            plt.ylabel("Northing [meters]",fontsize=FONTSIZE)
            plt.tick_params(axis="both", which="major", labelsize=FONTSIZE)
            plt.tick_params(axis="both", which="minor", labelsize=FONTSIZE)
            plt.legend(fontsize=FONTSIZE_LEGEND)
            plt.tight_layout()
            plt.grid(alpha=0.3, which="both")
            plt.savefig("example_positioning.png")
            plt.show()


    plt.figure()
    plt.plot(np.mean(err_arrays,axis=0), linewidth=2,  label='Positioning error w.r.t. groud truth')    
    plt.xlabel("Simulation steps [seconds]",fontsize=FONTSIZE)
    plt.ylabel("Positioning error [meters]",fontsize=FONTSIZE)
    plt.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    plt.tick_params(axis="both", which="minor", labelsize=FONTSIZE)
    plt.tight_layout()
    plt.grid(alpha=0.3, which="both")
    plt.savefig("benchmark_positioning_error.png")
    plt.show()

    '''
    plt.xlabel("Time [seconds]", fontsize=FONTSIZE)
    plt.ylabel("Heading error [degrees, unwrapped]",fontsize=FONTSIZE)
    plt.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    plt.tick_params(axis="both", which="minor", labelsize=FONTSIZE)
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.legend(fontsize=FONTSIZE_LEGEND)
    plt.savefig("compass_alpha_sweep.png",dpi=900)
    plt.show()
    '''