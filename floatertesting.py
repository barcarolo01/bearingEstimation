
from Floater import *


if __name__ == "__main__":
    STEPS = 900
    errs, psi_rms = [], []
    for seed in range(200):
        np.random.seed(seed)
        f = Floater(0, 0, 0, 0, 1, 1.0, rot_seed=10000+seed)
        f.set_initial_velocity(0.1, 0.1, 0)
        f.set_rho(0.999)
        f.set_sigma(0.2, 0.2, 0)
        for _ in range(STEPS):
            f.move()
        R = f.return_results()
        errs.append(np.linalg.norm(R['gt'][-1, :2] - R['imu'][-1, :2]))
        psi_rms.append(np.sqrt(np.mean(R['psi_err']**2)))

        if seed < 1:
            print(f"========== {seed} ==========")
            plt.figure()
            plt.plot(R['psi_err'],color='red')
            plt.show()

            '''
            plt.figure()
            plt.scatter(R['gt'][:,0],R['gt'][:,1],color='red',linewidths=0.1)
            plt.scatter(R['imu'][:,0],R['imu'][:,1],color='blue',linewidths=0.1)
            print(f"ERR DI POS MEDIO: {np.mean(np.linalg.norm(R['gt'][:,:2]-R['imu'][:,:2],axis = 1))}")
            plt.show()
            plt.figure()
            plt.title("POS ERROR")
            plt.plot(np.linalg.norm(R['gt'][:,:2]-R['imu'][:,:2],axis = 1))
            plt.show()
            '''

            

    print(f"errore finale: MEDIO {np.mean(errs):.1f} ")
    print(f"errore finale: mediana {np.median(errs):.1f} m, 90mo pct {np.percentile(errs, 90):.1f} m")
    print(f"heading RMS  : mediana {np.median(psi_rms):.2f} deg")