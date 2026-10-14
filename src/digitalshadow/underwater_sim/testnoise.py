from digitalshadow.underwater_sim.ambient_noise import add_ambient_noise
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    signal = np.zeros(96000)
    out,_ = add_ambient_noise([signal],96000,shipping=1)
    plt.plot(out[0])
    plt.savefig("PROVAA1.png")