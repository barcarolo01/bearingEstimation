import math
from Floater import *
import Floater
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
from Positioning.model import distance_emulation, movement_emulation
from Positioning.algo import estimate
from Positioning.decision import resurface_decision, mds_decision
from Positioning.reverse import reverse
from Positioning.error import estimate_mov_error

np.random.seed(256123)

# ===== FLAG DI PLOTTING =====
plot_f = True
plot_sim_f = True

# ===== SIMULAZION PARAMETERS =====
D = 3 # 3D or 2D
I = 1000 # Number of steps
N = 5 # Numero di devices

DIST_RATIO = 13 # Not used by Davide
RANGE = 100 # Random initial positions in [0, RANGE]
LOCAL_F = False # If True, distances are measured only w.r.t. reference node


def resurface(self_pos, self_err, real_pos, ref):
    """
    Node RESET
    
    Args:
        self_pos (ndarray): N x D matrix conteining the estimated positions
        self_err (ndarray): N x 1 vector of cumulated errors
        real_pos (ndarray): N x D matrix of real positions
        ref (int): index of the node to be reset
    
    Returns:
        tuple: (self_pos aggiornata, self_err aggiornato)
    """
    print("Resurface: ", ref, "\n")
    self_err[ref] = 0 # Erros is zeroed
    self_pos[ref, :] = real_pos[ref, :].copy() # Estimated position is set to the real one

    return self_pos, self_err

'''
def random_positions(n, d):
    pos = np.random.random([n, d]) * RANGE
    return pos


def random_movement(n, d):
    """
    Genera movimento casuale per N nodi in spazio d-dimensionale.
    
    Movimento è estratto uniformemente da [-MOV_RANGE/2, MOV_RANGE/2]
    per ogni coordinata, normalizzato per la dimensione.
    
    Args:
        n (int): numero di nodi
        d (int): dimensioni
    
    Returns:
        ndarray: matrice N x D con movimento casuale
    """

    MOV_RANGE = 30 # Movimento casuale in ogni step [-MOV_RANGE/2, MOV_RANGE/2]

    mov = (np.random.random([n, d]) - 0.5) * 2 * MOV_RANGE / sqrt(D)
    return mov
'''

def plot_positions(pos, color='blue'):
    """
    Plots 2D positions of nodes
    
    Args:
        pos (ndarray): N x D matrix of positions
        color (str): color (matplotlib)
    
    (?) First node is marked with a 'x', other with a circle
    """
    plt.scatter(pos[1:, 0], pos[1:, 1], c=color)
    plt.scatter(pos[0, 0], pos[0, 1], c=color, marker='x', label='_nolegend_')
    plt.xlim([-200,200])
    plt.ylim([-200,200])

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


if __name__ == "__main__":    
    # === IMU only ===
    self_plain_poss = []
    self_plain_errs = [] 

    # === IMU + MDS ===
    self_poss = [] 
    self_errs = []

    real_poss = []
    self_movs = []
    dists = [dict() for _ in range(N)]
    
    # Flaoter initialization
    floaters = []
    for i in range(N):
        dists.append(dict())

        # Creation of a new floater
        f = Floater(
            gt_x=np.random.uniform(0, 100), # Random coordinates
            gt_y=np.random.uniform(0, 100),
            gt_z=1000,  # Constant depth
            dt=1.0,
            dim=3
        )
        f.set_rho(0.95)
        f.set_sigma(0.3, 0.3, 0.0)
        floaters.append(f)

    # Initial positions
    real_pos = np.array([f.get_gt_position() for f in floaters])
    self_pos = real_pos.copy()
    self_plain_pos = real_pos.copy()

    # Arrays of cumulated errors
    self_err = np.zeros(N)
    self_plain_err = np.zeros(N)
    
    # Resurface counter
    resurface_tot = np.zeros(N, dtype=int)
    resurface_plain_tot = np.zeros(N, dtype=int)

    # Iteration over every simulation step
    for i in range(I):
        print(i)

        # Fetch the old ground truth and estimated positions
        old_gt_pos = np.array([f.get_gt_position() for f in floaters])
        old_est_pos = np.array([f.get_est_position() for f in floaters])

        for n in range(1):
            print(f"======== FLOATER {n} =======")
            print(f"old_gt_pos     {old_gt_pos[n]}")
            print(f"old_est_pos      {old_est_pos[n]}")
            print(f"self_plain_pos     {self_plain_pos[n]}")
            print()
          

        # === EVERY FLOATER MOVES ===
        for n in range(N):
            floaters[n].move()

        # Fetch the old ground truth and estimated positions
        new_real_pos = np.array([f.get_gt_position() for f in floaters])
        new_est_pos = np.array([f.get_est_position() for f in floaters])

        # Movement in the last iteration: diference between current and previous estimations
        self_mov = new_est_pos - old_est_pos        

        # Update positions and errors
        self_pos += self_mov
        self_plain_pos += self_mov
        self_err += estimate_mov_error(self_mov)
        self_plain_err += estimate_mov_error(self_mov)


        # (Evey 'RESURFACE_FREQ' iterations)
        if mds_decision(i, self_err, local=LOCAL_F):
            print(f"[MDS] launched ad iteration {i}")
            self_pos_copy, self_err_copy = self_pos.copy(), self_err.copy()
            
            # Iteration over the number of floaters
            for n in range(N):
                # Measure the distance between every possible couple of nodes (NxN matrix)
                measured_dist = distance_emulation(new_real_pos, local=LOCAL_F, ref=n)
                dists[n][i] = measured_dist  # Salva per retroazione backwar
                
                # TODO: CHECK! RUN (MDS + Procrustes)
                self_pos_new, self_err_new = estimate(self_pos_copy, self_err_copy, measured_dist)
                
                # Update only the floater currently under analysis: the others will be updated with their own MDS
                # Note: 'plain' position and error are not updated since they assume no MDS is run
                self_pos[n, :] = self_pos_new[n, :].copy()
                self_err[n] = self_err_new[n]

        
        # Iteration over every floaters
        for n in range(N):
            # i.e. every 'RESURFACE_FREQ' simulation steps
            if resurface_decision(self_err, i, n):
                # Restore the position and reser the error of the n-th floater (IMU+MDS variables)
                self_pos, self_err = resurface(self_pos, self_err, new_real_pos, n)
                resurface_tot[n] += 1
                floaters[n].resurface(new_real_pos[n])

        for n in range(N):
            if resurface_decision(self_plain_err, i, n):
                # Restore the position and reser the error of the n-th floater (IMU only variables)
                self_plain_pos, self_plain_err = resurface(self_plain_pos, self_plain_err, new_real_pos, n)
                resurface_plain_tot[n] += 1
                floaters[n].resurface(new_real_pos[n])



        self_poss.append(self_pos.copy()) # IMU+MDS positions
        self_plain_poss.append(self_plain_pos.copy()) # IMU only positions
        real_poss.append(new_real_pos.copy()) # ground truth
        self_movs.append(self_mov.copy()) # movement
        self_errs.append(self_err.copy()) # IMU+MDS error
        self_plain_errs.append(self_plain_err.copy()) # IMU only error

        
        # ----- VISUALIZZAZIONE PROGRESS =====
        if plot_sim_f:# and i % 10 == 0:
            '''
            print(f"  Iterazione {i}/{I}")
            print(f"real_pos[0]:\t\t {real_pos[0]}")
            print(f"self_plain_pos[0]:\t {self_plain_pos[0]}")
            print(f"self_pos[0]:\t\t {self_pos[0]}")
            '''
            plot_positions(new_real_pos, 'green')
            plot_positions(self_plain_pos, 'blue')
            plot_positions(self_pos, 'red')
            
            plt.legend(["true", "self (IMU only)", "estimated (IMU+MDS)"])
            plt.title(f"Node positions - Iteration {i}")
            #plt.show()
            plt.savefig(f"img/{i}.png")
            plt.close('all')
        

    # Save the "forward-only" status
    old_self_poss, old_self_plain_poss = self_poss.copy(), self_plain_poss.copy()
    old_self_errs, old_self_plain_errs = self_errs.copy(), self_plain_errs.copy()

    # Run MDS "reverse"
    self_poss, self_errs = reverse(self_movs, self_errs.copy(), self_poss.copy(), dists)
    
    # Run MDS "forward only"
    dists_empty = []
    for i in range(N):
        dists_empty.append(dict())
    self_plain_poss, self_plain_errs = reverse(self_movs, self_plain_errs.copy(), 
                                                self_plain_poss.copy(), dists_empty)

    # ===== PLOTTING
    if plot_f:
        # Plot 1: Cumulated error
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(compute_means(self_plain_errs), label="IMU only (reverse)", alpha=0.7)
        plt.plot(compute_means(self_errs), label="IMU + MDS (reverse)", alpha=0.7)
        plt.plot(compute_means(old_self_plain_errs), label="IMU only (forward)", alpha=0.7)
        plt.plot(compute_means(old_self_errs), label="IMU + MDS (forward)", alpha=0.7)
        plt.legend()
        plt.title("Self errors (accumulated error per node)")
        plt.xlabel("Iteration")
        plt.ylabel("Mean error")
        plt.grid(True, alpha=0.3)

        # Plot 2: Error vs ground truth
        plt.subplot(1, 2, 2)
        plt.plot(compute_error(real_poss, self_plain_poss), label="IMU only (reverse)", alpha=0.7)
        plt.plot(compute_error(real_poss, self_poss), label="IMU + MDS (reverse)", alpha=0.7)
        plt.plot(compute_error(real_poss, old_self_plain_poss), label="IMU only (forward)", alpha=0.7)
        plt.plot(compute_error(real_poss, old_self_poss), label="IMU + MDS (forward)", alpha=0.7)
        plt.legend()
        plt.title("Positioning errors (vs ground truth)")
        plt.xlabel("Iteration")
        plt.ylabel("Mean error")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        plt.close("all")


exit(0)