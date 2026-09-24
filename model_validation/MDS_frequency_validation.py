from digitalshadow.devices.Floater import *
from digitalshadow.devices.Transmitter import *
from simulation import *

SIMULATE = True

FONTSIZE = 14
FONTSIZE_LEGEND = 12

NUMBER_OF_FLOATERS = 6
N_SIMULATIONS = 10
STEPS = 3600
PERIODS = [60,120,180,300,600,900,1200,1800]

if __name__ == '__main__':
        Center = [32.839, -34.635]      
        periods_toplot = []
        errors_imu_toplot = []
        errors_imu_imu_toplot = []

        for period in PERIODS:
                imu_avg = np.zeros((N_SIMULATIONS,STEPS))
                imu_mds_avg = np.zeros((N_SIMULATIONS,STEPS))

                for seed in range(N_SIMULATIONS):               
                        TX = Transmitter(-1,0,0,0,1.0)
                        TX.set_initial_velocity(np.random.uniform(0.5),np.random.uniform(0.5),0)
                        TX.set_sigma(0.1,0.1,0)
                        TX.Rho = 0.999

                        floaters = []
                        for i in range(NUMBER_OF_FLOATERS):
                                f = Floater(ID = i,
                                                gt_x=np.random.uniform(1000)-500,
                                                gt_y=np.random.uniform(1000)-500,
                                                gt_z=10,
                                                NF=NUMBER_OF_FLOATERS,
                                                dt=1.0)

                                f.set_initial_velocity(np.random.uniform(1)-0.5, np.random.uniform(1)-0.5, 0.0)
                                f.set_sigma(np.random.uniform(0.1), np.random.uniform(0.1), 0.0)
                                f.use_compass = True
                                f.Rho = 0.999
                                f.Rho_yaw = 0.999
                                f.gt_omega = np.random.uniform(0.1)-0.05
                                f.MDS_freq = period
                                floaters.append(f)

                        sim = Simulation(Floaters=floaters,
                                        Steps=STEPS,
                                        Center=Center,
                                        transmitter=TX,
                                        seed=seed)

                        sim_result = sim.run_simulation()

                        imu_avg[seed] = np.mean(np.mean(sim_result['errors_imu'],axis=0))
                        imu_mds_avg[seed] = np.mean(np.mean(sim_result['errors_imu_mds'],axis=0))

                periods_toplot.append(period)
                errors_imu_toplot.append(np.mean(imu_avg))
                errors_imu_imu_toplot.append(np.mean(imu_mds_avg))


        periods_toplot = np.asarray(periods_toplot)
        imu_values = np.asarray(errors_imu_toplot)
        mds_values = np.asarray(errors_imu_imu_toplot)

        plt.figure(figsize=(15, 5))
        plt.plot(periods_toplot, np.ones(periods_toplot.shape)*np.mean(imu_values),'o-', label='IMU', color="#0000ff")
        plt.plot(periods_toplot, mds_values,'o-', label='IMU+MDS', color="#2AB040")

        # Percentuale di variazione IMU+MDS rispetto a IMU
        percent_changes = (mds_values - imu_values) / imu_values * 100

        # Etichette sotto i punti della seconda serie
        for x, y, percentage in zip(periods_toplot, mds_values, percent_changes):
                if x == 180:
                        plt.annotate(
                                f'{percentage:.1f}%',
                                (x, y),
                                textcoords="offset points",
                                xytext=(12, -12),
                                ha='center',
                                va='top'
                        )
                else:
                        plt.annotate(
                                f'{percentage:.1f}%',
                                (x, y),
                                textcoords="offset points",
                                xytext=(6, -12),
                                ha='center',
                                va='top'
                        )

        
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.legend(fontsize=FONTSIZE_LEGEND, loc='lower right')
        plt.tick_params(axis="both", which="major", labelsize=FONTSIZE)
        plt.tick_params(axis="both", which="minor", labelsize=FONTSIZE)
        plt.xticks(periods_toplot)
        plt.grid(axis='x',linestyle='--',alpha=0.5)
        plt.grid(axis='y',linestyle='--',alpha=0.5)
        plt.xlabel("MDS frequency [seconds]",fontsize=FONTSIZE)
        plt.ylabel("Average positioning error [meters]",fontsize=FONTSIZE)
        plt.tight_layout()
        plt.savefig("MDS_example.png", dpi=900)
        plt.show()