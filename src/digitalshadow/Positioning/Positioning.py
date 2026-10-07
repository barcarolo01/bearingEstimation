import numpy as np

DEF_WEIGHT = 0.00001  # Peso di default quando errore stima è infinito
ERR_TH = 0.99         # Accetta se errore_nuovo < 0.99 * errore_vecchio

def estimate(self_pos, self_err, dist, dist_w):
    # Replace all NaN with 0
    dist = np.nan_to_num(np.asarray(dist, dtype=float), nan=0.0)
    pos_w = mds_weights(self_pos, self_err, dist)

    estimated_pos = mds_algo(self_pos, dist, dist_w, pos_w)
    #estimated_pos = mds_algo_3d(self_pos, dist, dist_w, pos_w)  
    return update_decision(self_pos, estimated_pos, dist, self_err, dist_w)

def estimate_pos_error(pos, pos_new, dist, ref, mask):
    #new_dist = np.linalg.norm(pos - pos_new[ref, :], axis=1)
    new_dist = np.linalg.norm(pos_new - pos_new[ref, :], axis=1)
    new_dist[ref] = 0
    old_dist = np.linalg.norm(pos - pos[ref, :], axis=1)
    old_dist[ref] = 0

    new_diff = abs(new_dist - dist[ref, :])
    old_diff = abs(old_dist - dist[ref, :])

    m = mask[ref, :].astype(bool)
    new_diff, old_diff = new_diff[m], old_diff[m]

    den = np.sum(old_diff)
    if den == 0 or new_diff.size == 0:
        return 1.0                     # nessuna informazione: nessun miglioramento
    return np.sum(new_diff) / den

def update_decision(pos, estimated_pos, dist, err, mask):
    estimated_errors = np.zeros(np.shape(err)[0])

    for r in range(np.shape(err)[0]):
        estimated_errors[r] = estimate_pos_error(pos, estimated_pos, dist, r, mask)

    accept = estimated_errors < ERR_TH

    pos, err = pos.copy(), err.copy()
    pos[accept] = estimated_pos[accept]
    err[accept] = err[accept] * estimated_errors[accept]

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
        V[-1, :] *= -1  # Rifletti l'ultima riga di V
        R = np.dot(V.T, U.T)
    
    # TRANSLATION
    # t = centro_globale - R @ centro_locale
    t = np.transpose(center) - R.dot(np.transpose(center_local))

    return R, t

def mds_algo(pos, dist, dist_w, pos_w):
    """
    MDS 2D con profondità nota + allineamento di Procrustes.

    Le distanze misurate sono oblique: vengono proiettate sul piano
    orizzontale usando le profondità, che si assumono note esattamente.
    La z non viene mai stimata, solo riattaccata alla fine.

    Args:
        pos    (ndarray): N x 3, posizioni a priori (la colonna z è esatta)
        dist   (ndarray): N x N, distanze oblique misurate (metri)
        dist_w (ndarray): N x N, pesi / maschera di disponibilità
        pos_w  (ndarray): N x 1, pesi per l'allineamento

    Returns:
        ndarray: N x 3, posizioni stimate (xy dall'MDS, z misurata)
    """
    pos = np.asarray(pos, dtype=float)
    z   = pos[:, 2]

    # Proiezione sul piano orizzontale: il max protegge dal rumore
    # che renderebbe |dz| maggiore della distanza obliqua
    dz     = z[:, None] - z[None, :]
    dist_h = np.sqrt(np.maximum(dist**2 - dz**2, 0.0))

    # SMACOF in due dimensioni, seminato dalle posizioni correnti
    xy_MDS = weighted_smacof(dist_h, dist_w, init=pos[:, :2])

    # Allineamento rigido sul piano
    R, t = roto_trans(pos[:, :2], xy_MDS, pos_w)
    xy   = (R.dot(np.transpose(xy_MDS)) + np.reshape(t, (-1, 1))).T

    estimated_pos = np.column_stack([xy, z])

    # Nodi senza alcuna distanza nota: nessuna informazione, resta il priore
    isolated = (dist_w.sum(axis=1) == 0)
    if isolated.any():
        estimated_pos[isolated] = pos[isolated]

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
    # Replace all NaN with 0
    # ===== PESI POSIZIONI =====
    # Peso inversamente proporzionale all'errore del nodo
    pos_w = 1 / (err + 10e-9)
    pos_w[np.isinf(pos_w)] = DEF_WEIGHT
    pos_w = pos_w / np.sum(pos_w) * np.size(pos_w)
    
    # Reshape per operazioni matriciali
    pos_w = np.reshape(pos_w, [np.size(pos_w), 1])

    return pos_w

def weighted_smacof(D, W, init, n_iter=500, eps=1e-9):
    X = np.array(init, dtype=float)
    W = np.array(W, dtype=float); np.fill_diagonal(W, 0.0)
    D = np.nan_to_num(np.array(D, dtype=float), nan=0.0)

    V  = -W.copy(); np.fill_diagonal(V, W.sum(axis=1))
    Vp = np.linalg.pinv(V)

    for _ in range(n_iter):
        dX = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        ratio = np.divide(D, dX, out=np.zeros_like(D), where=dX > eps)
        B = -W * ratio
        np.fill_diagonal(B, 0.0)
        np.fill_diagonal(B, -B.sum(axis=1))
        X = Vp @ (B @ X)
    return X




def mds_algo_3d(pos, dist, dist_w, pos_w):
    """
    MDS 3D + allineamento di Procrustes (versione precedente alla
    proiezione sul piano orizzontale, tenuta per confronto).

    Args:
        pos    (ndarray): N x 3, posizioni a priori
        dist   (ndarray): N x N, distanze oblique misurate (metri)
        dist_w (ndarray): N x N, pesi / maschera di disponibilità
        pos_w  (ndarray): N x 1, pesi per l'allineamento

    Returns:
        ndarray: N x 3, posizioni stimate
    """
    pos = np.asarray(pos, dtype=float)

    pos_MDS = weighted_smacof(dist, dist_w, init=pos, n_iter=500)

    R, t = roto_trans(pos, pos_MDS, pos_w)
    estimated_pos = (R.dot(np.transpose(pos_MDS))+ np.reshape(t, (-1, 1))).T

    isolated = (dist_w.sum(axis=1) == 0)
    if isolated.any():
        estimated_pos[isolated] = pos[isolated]

    return estimated_pos