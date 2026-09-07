import numpy as np
from sklearn.manifold import MDS

DEF_WEIGHT = 0.00001  # Peso di default quando errore stima è infinito
ITER = 50            # Iterazioni MDS per convergenza
ERR_TH = 0.99         # Accetta se errore_nuovo < 0.99 * errore_vecchio
FIXED_DIST_ERR = 10  # Errore fisso su distanze (unità)

def estimate(self_pos, self_err, dist):
    # Compute MDS weights
    dist_w, pos_w = mds_weights(self_pos, self_err, dist)
    
    # Run MDS algorithm
    estimated_pos = mds_algo(self_pos, dist, dist_w, pos_w)

    # Step 4: Decide se accettare e aggiorna errori
    return update_decision(self_pos, estimated_pos, dist, self_err)

def estimate_pos_error(pos, pos_new, dist, ref):
    """
    Stima l'errore di una stima di posizione.
    
    ALGORITMO:
    Confronta le distanze misurate con le distanze calcolate da:
    1. Vecchia posizione (pos)
    2. Nuova posizione (pos_new)
    
    Vince la stima che ha distanze più coerenti con le misurazioni.
    
    Nota sulla normalizzazione:
    - new_diff: somma dei residui con nuova stima
    - old_diff: somma dei residui con vecchia stima
    - Ritorna: new_diff / old_diff (rapporto di miglioramento)
    
    Se rapporto < 1: nuova stima è migliore
    Se rapporto > 1: vecchia stima è migliore
    
    Args:
        pos (ndarray): matrice N x D delle VECCHIE posizioni
        pos_new (ndarray): matrice N x D delle NUOVE posizioni
        dist (ndarray): matrice N x N delle distanze misurate
        ref (int): indice del nodo di riferimento
    
    Returns:
        float: rapporto di errore (nuovo / vecchio)
    """
    
    # Calcola distanze dalla nuova stima rispetto a ref
    new_dist = np.linalg.norm(pos - pos_new[ref, :], axis=1)
    new_dist[ref] = 0  # Distanza da se stesso è 0
    
    # Calcola distanze dalla vecchia stima rispetto a ref
    old_dist = np.linalg.norm(pos - pos[ref, :], axis=1)
    old_dist[ref] = 0  # Distanza da se stesso è 0

    # Calcola errori (differenza tra calcolato e misurato)
    new_diff = abs(new_dist - dist[ref, :])
    old_diff = abs(old_dist - dist[ref, :])

    # Ritorna rapporto: nuovi errori / vecchi errori
    return np.sum(new_diff) / np.sum(old_diff)

def estimate_dist_error(dist):
    # TODO da rivedere
    """
    Stima l'errore nella misurazione di distanze.
    
    MODELLO SEMPLICE:
    Aggiunge un errore fisso (FIXED_DIST_ERR) a ogni distanza.
    
    Questo modella:
    - Errore di bias del sensore
    - Errore di multipath (riflessioni)
    - Incertezza nella misurazione
    
    Args:
        dist (ndarray): matrice N x N delle distanze misurate
    
    Returns:
        ndarray: matrice N x N dell'errore stimato
    """
    return dist + FIXED_DIST_ERR

def update_decision(pos, estimated_pos, dist, err):
    """
    Decide se accettare la nuova stima da MDS o mantenere la vecchia.
    
    ALGORITMO:
    1. Per ogni nodo, calcola l'errore della stima rispetto alle distanze
    2. Se l'errore medio migliora (rapporto < ERR_TH), accetta
    3. Aggiorna errore del nodo moltiplicando per il fattore di miglioramento
    
    NOTA IMPORTANTE:
    - estimated_errors è un rapporto (nuovo_errore / vecchio_errore)
    - Se rapporto < 1: nuovo errore è minore (migliore)
    - Se rapporto > 1: nuovo errore è maggiore (peggio)
    - Usiamo soglia 0.99 (accetta solo se errore diminuisce di >1%)
    
    Args:
        pos (ndarray): matrice N x D delle posizioni VECCHIE
        estimated_pos (ndarray): matrice N x D delle posizioni NUOVE (da MDS)
        dist (ndarray): matrice N x N delle distanze misurate
        err (ndarray): vettore N dell'errore accumulato VECCHIO
    
    Returns:
        tuple: (pos aggiornata, err aggiornato)
    """
    
    # Calcola errore per ogni nodo
    estimated_errors = np.zeros(np.shape(err)[0])
    for r in range(np.shape(err)[0]):
        estimated_errors[r] = estimate_pos_error(pos, estimated_pos, dist, r)

    # Se migliore della soglia, accetta la nuova stima
    if np.mean(estimated_errors) < ERR_TH:
        # Accetta nuove posizioni
        pos = estimated_pos.copy()
        # Aggiorna errori: err_nuovo = err_vecchio * rapporto_miglioramento
        err *= estimated_errors
    
    return pos, err

def roto_trans(pos, pos_MDS, weights):
    """
    The method computed rototranslation  parameters (R ant T) that best matches
    two cloud points  (pos and pos_MDS)
    
    Inputs:
        pos: matrix (N,D) of the position estimated by floaters
        pos_MDS: matrix (N,D) returned by MDS
        weights: array (N,) of weights

    Returns:
            R: rotation matrix (D,D)
            t: translation vector (D,)
    """

    center_local = np.sum(pos_MDS * weights, axis=0) / np.sum(weights)
    center = np.sum(pos * weights, axis=0) / np.sum(weights)

    
    directions_MDS = (pos_MDS - center_local)
    directions = (pos - center)

    # W is a diagonal matrix of weights
    W = np.diag(np.reshape(weights, np.size(weights)))

    # S = pos_MDS^T @ W @ pos (Weighted covariance matrix)
    S = np.transpose(directions_MDS).dot(W).dot(directions)

    # ===== SVD DECOMPOSITION =====
    U, _, V = np.linalg.svd(S)

    # ROTATION
    # Soluzione ottimale: R = V^T @ U^T
    R = np.dot(V.T, U.T)
    
    # ===== VERIFICA DETERMINANTE =====
    d = np.linalg.det(R)
    if d < 0:
        V[2, :] *= -1  # Rifletti l'ultima riga di V
        R = np.dot(V.T, U.T)
    
    # TRANSLATION
    # t = centro_globale - R @ centro_locale
    t = np.transpose(center) - R.dot(np.transpose(center_local))

    return R, t


def mds_algo(pos, dist, dist_w, pos_w):
    """
    Algoritmo completo di MDS + Procrustes.
    
    Flusso:
    1. Applica MDS per ricostruire posizioni locali da distanze
    2. Calcola trasformazione rigida (rotazione + traslazione) per allinearle
    3. Applica trasformazione per ottenere posizioni stimate nel frame globale
    
    Args:
        pos (ndarray): matrice N x D delle posizioni attuali (stimate/reali)
        dist (ndarray): matrice N x N delle distanze misurate
        dist_w (ndarray): matrice N x N dei pesi per distanze
        pos_w (ndarray): vettore N x 1 dei pesi per posizioni
    
    Returns:
        ndarray: matrice N x D delle posizioni stimate (allineate)
    """
    n = np.shape(pos)[1]  # Numero di dimensioni
        
    # Weight normalization (they sum up to 1)
    weights = dist_w / np.sum(dist_w)
    
    # MDS
    embedding = MDS(
        n_components=n,        # Maintain the same number of dimensions
        n_init=1,              # One initialization
        max_iter=ITER,         # Number of iterations
        eps=0,                 
        metric='precomputed',  # Dist is already a distance matrix
        init='classical_mds'
    )

    pos_MDS = embedding.fit_transform(dist, weights, init=pos)

    # Calculation of the maktranslation parameters
    R, t = roto_trans(pos, pos_MDS, pos_w)
    
    # Apply the rigid motion to obtain the estimated positions
    estimated_pos = R.dot(np.transpose(pos_MDS)) + np.reshape(t, [np.size(t), 1])
    estimated_pos = np.transpose(estimated_pos)

    return estimated_pos

def mds_weights(pos, err, dist):
    """
    Calcola i pesi per l'algoritmo MDS basati su errore stimato.
    
    LOGICA DEI PESI:
    
    Per le distanze:
    - Peso inversamente proporzionale all'errore stimato della distanza
    - Se dist[i,j] = 0 (non misurata): usa errore combinato dei due nodi
    - Pesi infiniti vengono rimpiazzati con DEF_WEIGHT
    - Normalizza in modo che somma dei pesi = numero di elementi
    
    Per le posizioni:
    - Peso inversamente proporzionale all'errore del nodo
    - Pesi infiniti vengono rimpiazzati con DEF_WEIGHT
    - Normalizza come sopra
    
    Args:
        pos (ndarray): matrice N x D delle posizioni
        err (ndarray): vettore N dell'errore accumulato per nodo
        dist (ndarray): matrice N x N delle distanze
    
    Returns:
        tuple: (dist_w matrice N x N, pos_w matrice N x 1) di pesi normalizzati
    """
    
    # ===== PESI DISTANZE =====
    # Stima l'errore su ogni distanza misurata
    dist_w = 1 / (estimate_dist_error(dist))
    
    # Per distanze non misurate (dist[i,j] = 0), usa errori dei nodi
    for i in range(0, np.shape(dist_w)[0]):
        for j in range(0, np.shape(dist_w)[1]):
            if dist[i][j] == 0 and i != j:
                # Errore della distanza = combinazione errori dei due nodi
                dist_w[i][j] = 1 / ((err[i] + err[j]) + 10e-9)
    
    # Sostituisci pesi infiniti con valore di default
    dist_w[np.isinf(dist_w)] = DEF_WEIGHT
    
    # Normalizza: somma = numero totale di elementi
    dist_w = dist_w / np.sum(dist_w) * np.size(dist_w)

    # ===== PESI POSIZIONI =====
    # Peso inversamente proporzionale all'errore del nodo
    pos_w = 1 / (err + 10e-9)
    pos_w[np.isinf(pos_w)] = DEF_WEIGHT
    pos_w = pos_w / np.sum(pos_w) * np.size(pos_w)
    
    # Reshape per operazioni matriciali
    pos_w = np.reshape(pos_w, [np.size(pos_w), 1])

    return dist_w, pos_w

def move_error(N, dt=1.0):
    """Sigma di posizione dopo N passi di sola odometria a partire da un'ancora."""
    from Floater import IMU_ACCEL_BIAS, IMU_SIGMA_WHITENOISE, IMU_SIGMA_BIAS_DRIVING

    if N <= 0:
        return 0.0
    sum2 = N*(N+1)*(2*N+1)/6.0
    sum3 = (N*(N+1)/2.0)**2
    sum4 = N*(N+1)*(2*N+1)*(3*N**2+3*N-1)/30.0
    s_bias  = np.abs(IMU_ACCEL_BIAS)       * dt**2 * N*(N+1)/2.0
    s_white = IMU_SIGMA_WHITENOISE          * dt**2 * np.sqrt(sum2)
    s_rw    = IMU_SIGMA_BIAS_DRIVING        * dt**2 * np.sqrt((sum4 + 2*sum3 + sum2)/4.0)
    return float(np.linalg.norm(np.sqrt(s_bias**2 + s_white**2 + s_rw**2)))
    #return float(np.sqrt(s_bias[0]**2 + s_white[0]**2 + s_rw[0]**2))


def _retrodict(pos_anchor, movs, k, a):
    """Posizione al passo k retrodetta dall'ancora a (a >= k) con sola odometria."""
    p = pos_anchor.copy()
    for t in range(a, k, -1):
        p = p - movs[t-1]          # movs[t-1] = spostamento da t-1 a t
    return p

def reverse_node(fl, pos_hist, err_hist, use_mds=True,
                 final_pos=None, gps_sigma=0.0, fuse=True,
                 nbr_hist_pos=None, nbr_hist_err=None):

    
    n, I, dt = fl.ID, fl.steps_counter, fl.dt
    movs = fl.estimated_movements

    smoothed, err_out = {}, {}
    end_pos = fl.gt_pos if final_pos is None else final_pos

    anchor_pos = {k: pos_hist[k].copy() for k in fl.Resurface_index}
    anchor_err = {k: 0.0 for k in fl.Resurface_index}
    anchor_pos[I] = np.asarray(end_pos, float).copy()
    anchor_err[I] = gps_sigma
    smoothed[I], err_out[I] = anchor_pos[I].copy(), gps_sigma

    frames = sorted(k for k in fl.MDS_index if k in fl.dist_matrices) if use_mds else []

    for k in reversed(frames):
        a = min(x for x in anchor_pos if x >= k)
        pos_bwd = _retrodict(anchor_pos[a], movs, k, a)
        err_bwd = float(np.hypot(anchor_err[a], move_error(a - k, dt)))

        pos_all = nbr_hist_pos[k].copy()          # stime dei vicini al frame k
        err_all = nbr_hist_err[k].copy()
        pos_all[n] = pos_bwd                       # la MIA riga: retrodetta dall'ancora
        err_all[n] = err_bwd

        new_pos, new_err = estimate(pos_all, err_all, fl.dist_matrices[k])

        # update_decision ha gia' rifiutato se i residui non migliorano
        smoothed[k] = new_pos[n].copy()
        err_out[k]  = float(new_err[n])

        anchor_pos[k], anchor_err[k] = smoothed[k].copy(), err_out[k]

    cur_pos = anchor_pos[I].copy()
    cur_err = anchor_err[I]
    cur_ref = I

    
    for k in range(I - 1, -1, -1):
        if k in smoothed:                      # frame MDS gia' risolto
            cur_pos, cur_err, cur_ref = smoothed[k].copy(), err_out[k], k
            continue
        if k in anchor_pos:                    # resurface: ancora forte
            smoothed[k], err_out[k] = anchor_pos[k].copy(), anchor_err[k]
            cur_pos, cur_err, cur_ref = smoothed[k].copy(), anchor_err[k], k
            continue

        pos_bwd = _retrodict(cur_pos, movs, k, cur_ref)
        err_bwd = float(np.hypot(cur_err, move_error(cur_ref - k, dt)))
        err_fwd = float(err_hist[k])

        if err_bwd < err_fwd:
            smoothed[k], err_out[k] = pos_bwd, err_bwd
        else:
            smoothed[k], err_out[k] = pos_hist[k].copy(), err_fwd
            cur_pos, cur_err, cur_ref = pos_hist[k].copy(), err_fwd, k   # RESET
            
    return smoothed, err_out