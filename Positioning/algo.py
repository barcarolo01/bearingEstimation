"""
ALGORITMI DI LOCALIZZAZIONE
============================

Questo modulo contiene gli algoritmi core per la localizzazione:
1. MDS (Multidimensional Scaling): ricostruisce posizioni da distanze
2. Procrustes: allinea due cloud di punti (rotazione + traslazione)
3. Trilateration: localizzazione basata su distanze da punti noti

Flusso principale:
  - Applica MDS per ottenere "forma" dal vincolo di distanze
  - Applica Procrustes per allineare con il frame di riferimento
  - Calcola pesi basati su errore stimato
"""

import numpy as np
from sklearn.manifold import MDS
from Positioning.error import estimate_dist_error, fill_distances
from Positioning.decision import update_decision


# ===== PARAMETRI =====
DEF_WEIGHT = 0.00001  # Peso di default quando errore stima è infinito
ITER = 50            # Iterazioni MDS per convergenza


def roto_trans(pos, pos_local, weights):
    """
    Calcola la migliore rotazione e traslazione tra due cloud di punti.
    
    PROBLEMA DI PROCRUSTES:
    Dato due set di punti:
      - pos: posizioni nel frame globale (pos_reale o estimata)
      - pos_local: posizioni nel frame locale (da MDS)
      - weights: peso di ogni punto
    
    Trova (R, t) che minimizzano: sum_i w_i * ||pos_i - (R @ pos_local_i + t)||^2
    
    Algoritmo (SVD-based):
    1. Centra entrambi i cloud usando media pesata
    2. Calcola matrice di covarianza pesata: S = pos_local^T @ W @ pos
    3. Scompone S = U @ Sigma @ V^T (SVD)
    4. Ottiene R = V^T @ U^T
    5. Verifica determinante (deve essere +1 per rotazione, non riflessione)
    6. Calcola traslazione: t = centro_globale - R @ centro_locale
    
    Args:
        pos (ndarray): matrice N x D delle posizioni globali
        pos_local (ndarray): matrice N x D delle posizioni locali (da MDS)
        weights (ndarray): vettore N di pesi
    
    Returns:
        tuple: (R matrice D x D di rotazione, t vettore D di traslazione)
    """
    # ===== CALCOLA CENTRI PESATI =====
    center_local = np.sum(pos_local * weights, axis=0) / np.sum(weights)
    center = np.sum(pos * weights, axis=0) / np.sum(weights)

    # ===== CENTRA I PUNTI =====
    directions_local = (pos_local - center_local)
    directions = (pos - center)

    # ===== COSTRUISCE MATRICE DI COVARIANZA PESATA =====
    # W è matrice diagonale dei pesi
    W = np.diag(np.reshape(weights, np.size(weights)))
    # S = pos_local^T @ W @ pos (matrice di covarianza pesata)
    S = np.transpose(directions_local).dot(W).dot(directions)

    # ===== SVD DECOMPOSITION =====
    U, _, V = np.linalg.svd(S)

    # ===== CALCOLA ROTAZIONE =====
    # Soluzione ottimale: R = V^T @ U^T
    R = np.dot(V.T, U.T)
    
    # ===== VERIFICA DETERMINANTE =====
    # det(R) deve essere 1 per rotazione pura
    # Se det = -1, è una riflessione (non vogliamo) -> correggi
    d = np.linalg.det(R)
    if d < 0:
        V[2, :] *= -1  # Rifletti l'ultima riga di V
        R = np.dot(V.T, U.T)
    
    # ===== CALCOLA TRASLAZIONE =====
    # t = centro_globale - R @ centro_locale
    t = np.transpose(center) - R.dot(np.transpose(center_local))

    return R, t


def apply_rt(R, t, pos_local):
    """
    Applica trasformazione rigida (rotazione + traslazione) a punti.
    
    Trasforma posizioni dal frame locale al frame globale:
    pos_globale = R @ pos_locale + t
    
    Args:
        R (ndarray): matrice D x D di rotazione
        t (ndarray): vettore D di traslazione
        pos_local (ndarray): matrice N x D delle posizioni locali
    
    Returns:
        ndarray: matrice N x D delle posizioni globali trasformate
    """
    # Applica R @ pos_local^T + t
    # Nota: R.dot(np.transpose(...)) = R @ pos_local^T
    estimated_pos = R.dot(np.transpose(pos_local)) + np.reshape(t, [np.size(t), 1])
    
    # Trasponi per tornare a formato N x D
    return np.transpose(estimated_pos)


def mds(pos, dist, weights):
    """
    Applica Multidimensional Scaling (MDS).
    
    MDS è un algoritmo di riduzione dimensionale che:
    - Input: matrice di distanze D (n x n)
    - Output: posizioni in spazio n-dimensionale che rispettano le distanze
    
    Usa sklearn.manifold.MDS con:
    - dissimilarity='precomputed': dist è già una matrice di distanze
    - weights: pesa ogni punto in base all'errore stimato
    - init=pos: inizializza con le posizioni attuali (accelera convergenza)
    
    Args:
        pos (ndarray): matrice N x D delle posizioni attuali (usate come init)
        dist (ndarray): matrice N x N delle distanze (dissimilarità)
        weights (ndarray): vettore N dei pesi
    
    Returns:
        ndarray: matrice N x D delle posizioni ricostruite
    """
    n = np.shape(pos)[1]  # Numero di dimensioni
    
    # Normalizza pesi a somma 1
    weights = weights / np.sum(weights)
    
    # Crea e applica MDS
    embedding = MDS(
        n_components=n,        # Mantieni numero di dimensioni
        n_init=5,              # Un'inizializzazione (usiamo quella custom)
        max_iter=ITER,         # Numero iterazioni
        eps=0,                 # Tolerance di convergenza
        metric='precomputed',  # dist è matrice di distanze, non raw data,
        init='classical_mds'
    )

    # Applica MDS con inizializzazione custom
    local_pos = embedding.fit_transform(dist, weights, init=pos)

    return local_pos


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
    # Step 1: MDS - ricostruisce forma da distanze
    pos_local = mds(pos, dist, dist_w)
    
    # Step 2: Procrustes - calcola allineamento ottimale
    R, t = roto_trans(pos, pos_local, pos_w)
    
    # Step 3: Applica trasformazione
    estimated_pos = apply_rt(R, t, pos_local)

    return estimated_pos


def prob_trilateration_algo(pos, err, dist, ref=0):
    """
    PLACEHOLDER: Algoritmo di trilateration probabilistica.
    
    Localizza nodi usando distanze da punti di riferimento noti.
    Non implementato nel codice fornito.
    
    Args:
        pos: posizioni attuali
        err: errori stimati
        dist: distanze misurate
        ref: nodo di riferimento
    
    Returns:
        Posizioni stimate
    """
    pass


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


def estimate(self_pos, self_err, dist):
    """
    FUNZIONE PRINCIPALE: Stima posizioni usando MDS.
    
    Flusso:
    1. Calcola pesi basati su errore accumulato
    2. Completa matrice distanze (riempie zeri con distanze calcolate)
    3. Applica MDS + Procrustes
    4. Decide se accettare le nuove posizioni
    
    Args:
        self_pos (ndarray): matrice N x D delle posizioni attuali
        self_err (ndarray): vettore N dell'errore accumulato
        dist (ndarray): matrice N x N delle distanze misurate
    
    Returns:
        tuple: (self_pos aggiornata, self_err aggiornato)
    """
    # Step 1: Calcola pesiì
    dist_w, pos_w = mds_weights(self_pos, self_err, dist)
    
    # Step 2: Completa distanze
    #dist_full = fill_distances(self_pos, dist)
    dist_full = dist

    # Step 3: Applica MDS
    estimated_pos = mds_algo(self_pos, dist_full, dist_w, pos_w)

    # Step 4: Decide se accettare e aggiorna errori
    return update_decision(self_pos, estimated_pos, dist_full, self_err)
