from matplotlib import pyplot as plt
import numpy as np
from Positioning.error import *
from Positioning.reverse import *
from Positioning.algo import *
from Positioning.model import *

MDS_FREQ = 9999

def compute_error(poss, estimationss):
    """
    Compute the average error for each frame as the euclidean norm of the difference
    between real position vector and estimated position vector.

    Args:
        poss (list): list of I matrices (real positions of shape N x D)
        estimationss (list): list of I matrices (estimated positions of shape N x D)
    
    Returns:
        list: list of I values (average error per frame)
    """
    res = []
    for i in range(len(poss)):
        pos, estimations = poss[i], estimationss[i]
        diff = np.linalg.norm(pos - estimations, axis=1)
        res.append(np.average(diff))
    return res

def compute_means(errs):
    """
    Calcola the average of the errors.

    Args:
        errs (list): list of vectors of shape (I,) of errors (each of N values)
    
    Returns:
        list: list of I values (average error per frame)
    """
    res = []
    for i in range(len(errs)):
        err = errs[i]
        res.append(np.mean(err))
    return res

class PositioningFramework:
    def __init__(self, RESURFACE_FREQ, init_position_matrix):
        self.RESURFACE_FREQ = RESURFACE_FREQ

        # Previous position
        self.previous_est_position_matrix = init_position_matrix.copy()

        # Current ground truth position
        self.real_pos = init_position_matrix.copy()

        # Current estimated position (IMU + MDS)
        self.self_pos = init_position_matrix.copy()

        # Current estimated position (IMU only)
        self.self_plain_pos = init_position_matrix.copy()


        self.N = init_position_matrix.shape[0]
        self.self_err = np.zeros(self.N)
        self.self_plain_err = np.zeros(self.N)
        self.steps_counter = 0
        self.dists = [dict() for _ in range(self.N)]

        # IMU only
        self.self_plain_poss = []
        self.self_plain_errs = [] 
    
        # IMU + MDS
        self.self_poss = [] 
        self.self_errs = []
    
        self.real_poss = []
        self.self_movs = []

    def update_positioning(self,new_gt_position_matrix,new_est_position_matrix):
        self.steps_counter += 1

        # Compute the estimated motion
        self.self_mov = new_est_position_matrix - self.previous_est_position_matrix
        self.previous_est_position_matrix = new_est_position_matrix.copy()
        self.real_pos = new_gt_position_matrix.copy()
        

        self.self_pos += self.self_mov
        self.self_plain_pos += self.self_mov
        self.self_err += estimate_mov_error(self.self_mov)
        self.self_plain_err += estimate_mov_error(self.self_mov)

        # === Append current status to history ===
        self.self_poss.append(self.self_pos.copy()) # IMU+MDS positions
        self.self_plain_poss.append(self.self_plain_pos.copy()) # IMU only positions
        self.real_poss.append(new_gt_position_matrix.copy()) # ground truth
        self.self_movs.append(self.self_mov.copy()) # movement
        self.self_errs.append(self.self_err.copy()) # IMU+MDS error
        self.self_plain_errs.append(self.self_plain_err.copy()) # IMU only error

        if self.steps_counter % self.RESURFACE_FREQ == 0:
            self.resurface()

        if self.steps_counter % MDS_FREQ == 0:
            self.run_MDS()

    def resurface(self):
        for n in range(self.N):
            print("Resurface: ", n, "\n")
            self.self_err[n] = 0 # Erros is zeroed
            self.self_pos[n, :] = self.real_pos[n, :].copy() # Estimated position is set to the real one

        for n in range(self.N):
            print("Resurface: ", n, "\n")
            self.self_plain_err[n] = 0 # Erros is zeroed
            self.self_plain_pos[n, :] = self.real_pos[n, :].copy() # Estimated position is set to the real one

    def run_MDS(self):
        print(f"[MDS] launched ad iteration {self.steps_counter}")
        self_pos_copy, self_err_copy = self.self_pos.copy(), self.self_err.copy()
        
        # Iteration over the number of floaters
        for n in range(self.N):
            # Measure the distance between every possible couple of nodes (NxN matrix)
            measured_dist = distance_emulation(self.real_pos, local=False, ref=n)
            self.dists[n][self.steps_counter] = measured_dist  # Salva per retroazione backwar
            
            # TODO: CHECK! RUN (MDS + Procrustes)
            self_pos_new, self_err_new = estimate(self_pos_copy, self_err_copy, measured_dist)
            
            # Update only the floater currently under analysis: the others will be updated with their own MDS
            # Note: 'plain' position and error are not updated since they assume no MDS is run
            self.self_pos[n, :] = self_pos_new[n, :].copy()
            self.self_err[n] = self_err_new[n]


    
    def end_simulation(self):
        # Save the "forward-only" status
        old_self_poss, old_self_plain_poss = self.self_poss.copy(), self.self_plain_poss.copy()
        old_self_errs, old_self_plain_errs = self.self_errs.copy(), self.self_plain_errs.copy()
    
        # Run MDS "reverse"
        self.self_poss, self.self_errs = reverse(self.self_movs, self.self_errs.copy(), self.self_poss.copy(), self.dists)
        
        # Run MDS "forward only"
        dists_empty = []
        for i in range(self.N):
            dists_empty.append(dict())
        self.self_plain_poss, self.self_plain_errs = reverse(self.self_movs, self.self_plain_errs.copy(), 
                                                             self.self_plain_poss.copy(), dists_empty)


        old_mean = np.mean([np.mean(e) for e in old_self_errs])
        new_mean = np.mean([np.mean(e) for e in self.self_errs])
        
        print(f"Forward mean error: {old_mean:.4f}")
        print(f"Reverse mean error: {new_mean:.4f}")
        
        if new_mean > old_mean:
            print("⚠️ WARNING: Reverse is worse than forward!")

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(compute_means(self.self_plain_errs), label="IMU only (reverse)", alpha=0.7)
        plt.plot(compute_means(self.self_errs), label="IMU + MDS (reverse)", alpha=0.7)
        plt.plot(compute_means(old_self_plain_errs), label="IMU only (forward)", alpha=0.7)
        plt.plot(compute_means(old_self_errs), label="IMU + MDS (forward)", alpha=0.7)
        plt.legend()
        plt.title("Self errors (accumulated error per node)")
        plt.xlabel("Iteration")
        plt.ylabel("Mean error")
        plt.grid(True, alpha=0.3)

        # Plot 2: Error vs ground truth
        plt.subplot(1, 2, 2)
        plt.plot(compute_error(self.real_poss, self.self_plain_poss), label="IMU only (reverse)", alpha=0.7)
        plt.plot(compute_error(self.real_poss, self.self_poss), label="IMU + MDS (reverse)", alpha=0.7)
        plt.plot(compute_error(self.real_poss, old_self_plain_poss), label="IMU only (forward)", alpha=0.7)
        plt.plot(compute_error(self.real_poss, old_self_poss), label="IMU + MDS (forward)", alpha=0.7)
        plt.legend()
        plt.title("Positioning errors (vs ground truth)")
        plt.xlabel("Iteration")
        plt.ylabel("Mean error")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        #plt.show()
        plt.savefig("miafigura.png")
        plt.close("all")

        return np.asarray(old_self_poss), np.asarray(old_self_plain_poss), np.asarray(self.self_poss), np.asarray(self.self_plain_poss)