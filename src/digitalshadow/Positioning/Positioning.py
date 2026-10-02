import numpy as np

DEF_WEIGHT = 0.00001  # Peso di default quando errore stima è infinito
ITER = 50            # Iterazioni MDS per convergenza
ERR_TH = 0.99         # Accetta se errore_nuovo < 0.99 * errore_vecchio

def estimate(self_pos, self_err, dist, dist_w):
    # Replace all NaN with 0
    dist = np.nan_to_num(np.asarray(dist, dtype=float), nan=0.0)
    pos_w = mds_weights(self_pos, self_err, dist)

    estimated_pos = mds_algo(self_pos, dist, dist_w, pos_w)
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

    print(f"  ratio per nodo: {np.round(estimated_errors, 4)}")
    print(f"  accettati: {np.where(accept)[0]}")
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
    '''
    embedding = MDS(
        n_components=n,        # Maintain the same number of dimensions
        n_init=5,              # One initialization
        max_iter=ITER,         # Number of iterations
        eps=0,                 
        metric='precomputed',  # Dist is already a distance matrix
        init='classical_mds'
    )
    pos_MDS = embedding.fit_transform(dist, weights, init=pos)
    '''
    pos_MDS = weighted_smacof(dist,weights,init=pos,n_iter=500)

    # Calculation of the maktranslation parameters
    R, t = roto_trans(pos, pos_MDS, pos_w)
    
    # Apply the rigid motion to obtain the estimated positions
    estimated_pos = R.dot(np.transpose(pos_MDS)) + np.reshape(t, [np.size(t), 1])
    estimated_pos = np.transpose(estimated_pos)
    
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

def weighted_smacof(D, W, init, n_iter=300, eps=1e-9):
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