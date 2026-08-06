"""
STIMA DEGLI ERRORI
==================

Questo modulo contiene funzioni per stimare l'errore accumulato dai sensori.

L'errore è cruciale perché:
1. Viene usato per calcolare i pesi negli algoritmi MDS
2. Viene usato per decidere quando fare resurface/reset
3. Viene usato per valutare se accettare o rifiutare stime MDS

Tipi di errore:
- Errore di movimento: da IMU (accumula nel tempo)
- Errore di posizione: deviazione dalla stima effettiva
- Errore di distanza: da sensori di distanza
"""

import numpy as np

# ===== PARAMETRI =====
FIXED_DIST_ERR = 10  # Errore fisso su distanze (unità)


def estimate_mov_error(mov):
    """
    Stima l'errore accumulato dal movimento misurato da IMU.
    
    LOGICA SEMPLICE:
    L'errore di dead-reckoning (integrazione del movimento) cresce
    con la velocità. Usiamo la norma euclidea del movimento come stima.
    
    Interpretazione:
    - Se il nodo si muove di poco: errore piccolo
    - Se il nodo si muove molto: errore grande
    - Questo è un semplice modello (realistico per IMU accurati)
    
    Args:
        mov (ndarray): matrice N x D del movimento (velocità)
    
    Returns:
        ndarray: vettore N dell'errore per ogni nodo
    """
    # Norma euclidea del movimento per ogni nodo
    return np.linalg.norm(mov, axis=1)


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


def fill_distances(pos, dist):
    """
    Completa la matrice di distanze dove mancano misurazioni.
    
    LOGICA:
    Se una distanza non è stata misurata (dist[i,j] = 0),
    calcola il valore dalla posizione stimata attuale.
    
    Questo permette a MDS di lavorare con matrice completa,
    anche se non tutte le distanze sono state misurate.
    
    Nota IMPORTANTE:
    Questa è un'approssimazione! Usa la stima attuale come "truth",
    che potrebbe contenere errore. Idealmente, MDS dovrebbe
    pesare di meno le distanze calcolate rispetto a quelle misurate.
    
    Args:
        pos (ndarray): matrice N x D delle posizioni attuali (stimate)
        dist (ndarray): matrice N x N delle distanze misurate
    
    Returns:
        ndarray: matrice N x N con distanze completate
    """
    for i in range(0, np.shape(dist)[0]):
        for j in range(0, np.shape(dist)[1]):
            if dist[i][j] == 0:
                # Se distanza non misurata, calcola da posizioni attuali
                dist[i][j] = np.linalg.norm(pos[i] - pos[j])

    return dist
