from digitalshadow.devices.Floater import *
from digitalshadow.devices.IMU_models import load_imu_model
from digitalshadow.devices.Transmitter import *
from simulation import *

FONTSIZE = 14
FONTSIZE_LEGEND = 12

if __name__ == '__main__':
        NUMBER_OF_SIMS = 30
        STEPS = 1800

        IMU_avg_errs = np.zeros((NUMBER_OF_SIMS,STEPS))
        IMU_MDS_avg_errs = np.zeros((NUMBER_OF_SIMS,STEPS))
        IMU_COMP_avg_errs = np.zeros((NUMBER_OF_SIMS,STEPS))

        random_sim = np.random.default_rng(256123)
        

        for k in range(NUMBER_OF_SIMS):
                NUMBER_OF_FLOATERS = 10
                
                #Center = [20.832813, 88.698390] # India, low depth
                Center = [32.839, -34.635] 

                
                TX = Transmitter(-1,0,0,0,1.0)
                TX.Rho = 0.999
                TX.set_initial_velocity(3,3,0)
                TX.set_sigma(0.1,0.1,0)
                

                #TX = compute_TX_circle_trajectory(10,0,350,35,200,True)

                # === Floater initialization ===
                floaters = []
                for i in range(NUMBER_OF_FLOATERS):
                        f = Floater(ID = i,
                                        gt_x=random_sim.uniform(-500,500),
                                        gt_y=random_sim.uniform(-500,500),
                                        gt_z=10,
                                        NF=NUMBER_OF_FLOATERS,
                                        dt=1.0)

                        f = load_imu_model(f,'ADIS16470',dt=1.0)
                        f.set_initial_velocity(random_sim.uniform(-0.5,0.5), random_sim.uniform(-0.5,0.5), 0.0)
                        f.set_sigma(random_sim.uniform(0.05), random_sim.uniform(0.05), 0.0)
                        f.Rho = 0.9999 + random_sim.uniform(0,0.0001)
                        f.Rho_yaw = 1.0
                        f.use_compass = True

                        #f.gyro_bias = 0
                        #f.sigma_gyro_bias = 0
                        #f.sigma_gyro_white_noise = 0
                        #f.sigma_gyro_bias_driving = 0
                        #f.compass_bias = 0
                        #f.compass_bias_sigma = 0
                        f.gt_omega = random_sim.uniform(0.1)-0.05
                        f.MDS_freq = 120
                        floaters.append(f)
                
                sim = Simulation(
                                transmitter=TX,
                                Floaters=floaters,
                                Steps=STEPS,
                                Center=Center,
                                #transmitter_coordinates=compute_TX_circle_trajectory(10,0,350,35,200),
                                seed=k+100)

                res = sim.run_simulation()

                RX_gt = res["RX_gt_pos"]
                RX_IMU = res["RX_imu"]
                RX_IMU_MDS = res["RX_imu_mds"]
                RX_IMU_COMP = res["RX_imu_comp"]

                diff = np.linalg.norm(RX_gt[:,:,:2]-RX_IMU[:,:,:2],axis = 2)
                imu_avg_error =  np.sqrt(np.mean(diff**2,axis=1))
                imu_RMSE = np.sqrt(np.mean(imu_avg_error**2))

                diff = np.linalg.norm(RX_gt[:,:,:2]-RX_IMU_MDS[:,:,:2],axis = 2)
                imu_mds_avg_error = np.sqrt(np.mean(diff**2,axis=1))
                imu_mds_RMSE = np.sqrt(np.mean(imu_mds_avg_error**2))

                diff = np.linalg.norm(RX_gt[:,:,:2]-RX_IMU_COMP[:,:,:2],axis = 2)
                imu_comp_avg_error = np.sqrt(np.mean(diff**2,axis=1))
                imu_comp_RMSE = np.sqrt(np.mean(imu_comp_avg_error**2))

                IMU_avg_errs[k,:] = imu_avg_error.copy()
                IMU_MDS_avg_errs[k,:] = imu_mds_avg_error.copy()
                IMU_COMP_avg_errs[k,:] = imu_comp_avg_error.copy()



        
        plt.figure(figsize=(8,6))
        imu_tobeplotted = np.sqrt(np.mean(IMU_avg_errs**2,axis=0)) 
        imu_mds_tobeplotted = np.sqrt(np.mean(IMU_MDS_avg_errs**2,axis=0)) 
        imu_comp_tobeplotted = np.sqrt(np.mean(IMU_COMP_avg_errs**2,axis=0))
        print()
        print(np.mean(imu_tobeplotted))
        print(np.mean(imu_mds_tobeplotted))
        print(np.mean(imu_comp_tobeplotted))
        plt.plot(imu_tobeplotted,color = "#0000FF",label="IMU")
        plt.plot(imu_mds_tobeplotted,color = "#2AB040",label="IMU+MDS")
        plt.plot(imu_comp_tobeplotted,color = "#D6940F",label="IMU compensated")
        plt.xlabel("Time [seconds]",fontsize=FONTSIZE)
        plt.ylabel("Positioning error [meters]",fontsize=FONTSIZE)
        plt.tick_params(axis="both", which="major", labelsize=FONTSIZE)
        plt.tick_params(axis="both", which="minor", labelsize=FONTSIZE)
        plt.grid(alpha=0.3, which="both")
        plt.tight_layout()
        plt.legend(loc="upper left",fontsize=FONTSIZE_LEGEND,frameon=True)
        plt.savefig("Positioning_error.png",dpi=900)
        plt.show()