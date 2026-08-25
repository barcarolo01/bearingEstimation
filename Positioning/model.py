"""
MODELLO DI SENSORI - EMULAZIONE DI ERRORI
==========================================

Questo modulo emula il comportamento realistico dei sensori:
- IMU: misura movimento con errore
- Distance Sensor: misura distanze tra nodi con errore relativo + fisso

L'errore è modellato con distribuzioni gaussiane.
"""

import numpy as np
import scipy.stats as sps

# ===== PARAMETRI ERRORE SENSORI =====
ERR_P = 0.95               # Probabilità (95%ile) per il distribution normale
e_imu = 0.2                # Errore IMU relativo: 20% della norma del movimento
e_measure = 0.01           # Errore distanza relativo: 1% della distanza misurata
e_measure_fix = 0.05       # Errore distanza fisso: ±5 unità (sempre presente)


def compute_target_sd(target, prob):
    """
    Converte una percentuale di errore in deviazione standard gaussiana.
    
    Dato che vogliamo ±target con probabilità prob (es. 95%),
    usiamo la funzione inversa della CDF normale (PPF).
    
    Esempio: target=0.01, prob=0.95
      - Vogliamo che nel 95% dei casi l'errore sia ±1%
      - Questo corrisponde a 1.96 deviazioni standard per distribuzione N(0,1)
      - Quindi sd_reale = 0.01 / 1.96 ≈ 0.0051
    
    Args:
        target (float): errore desiderato (es. 0.01 = 1%)
        prob (float): probabilità (es. 0.95 = 95%)
    
    Returns:
        float: deviazione standard della gaussiana
    """
    dist = sps.norm(loc=0, scale=1)
    def_target = dist.ppf(prob)  # Inverso di CDF: ppf(0.95) ≈ 1.96
    
    return target / def_target


def distance_emulation(pos, local, ref=None):
    """
    Emula misurazioni di distanza tra nodi con errore realistico.
    
    Due tipi di errore:
    1. Errore relativo: ±e_measure * distanza (percentuale della misura)
    2. Errore fisso: ±e_measure_fix (errore costante, es. bias sensore)
    
    Nota sulla topologia:
    - Se local=False: misura tutte le distanze (matrice completa)
    - Se local=True: misura solo distanze che coinvolgono il nodo ref
    
    Args:
        pos (ndarray): matrice N x D delle posizioni reali
        local (bool): se True, misura solo locale attorno a ref
        ref (int): indice del nodo di riferimento (se local=True)
    
    Returns:
        ndarray: matrice N x N simmetrica delle distanze misurate
    """
    n = np.shape(pos)[0]
    dist = np.zeros([n, n])
    mask = np.zeros([n, n])

    # ===== CALCOLA DISTANZE REALI =====
    for i in range(n):
        for j in range(i):
            # Se local=True, misura solo se i o j è il nodo ref
            if local and (i != ref and j != ref):
                continue
            # Distanza euclidea tra nodo i e j
            dist[i][j] = np.linalg.norm(pos[i] - pos[j])
            mask[i][j] = 1  # Marca quali distanze sono misurate

    # ===== AGGIUNGE ERRORI GAUSSIANI =====
    
    rand_dist_emu = np.random.default_rng(67890)
    
    # Errore relativo (dipende dalla distanza)
    sd_dist_e = compute_target_sd(e_measure, ERR_P)
    dist_deviation = rand_dist_emu.standard_normal((n, n)) * sd_dist_e
    dist_error = dist * dist_deviation  # Errore proporzionale alla distanza
    
    # Errore fisso (indipendente dalla distanza)
    sd_dist_e_fix = compute_target_sd(e_measure_fix, ERR_P)
    dist_fix_deviation = rand_dist_emu.standard_normal((n, n)) * sd_dist_e_fix
    dist_error += dist_fix_deviation  # Aggiungi errore fisso
    
    # Applica errore solo alle distanze misurate (maschera)
    dist_error *= mask
    
    # Distanze misurate = distanze reali + errori
    self_dist = dist + dist_error
    

    self_dist = dist
    # Rendi matrice simmetrica (se misuro dist(i,j), conosco dist(j,i))
    return self_dist + np.transpose(self_dist)


def movement_emulation(mov):
    """
    Emula il movimento misurato da un IMU con errore.
    
    L'IMU misura l'accelerazione e integra per ottenere il movimento.
    L'errore accumulato dipende dalla norma del movimento (velocità).
    
    Strategia di errore:
    1. Genera errore gaussiano casuale di norma unitaria
    2. Scala alla norma del movimento reale
    3. Applica fattore di errore relativo (e_imu = 20%)
    
    Questo modella errori sistematici che crescono con il movimento.
    
    Args:
        mov (ndarray): matrice N x D del movimento reale
    
    Returns:
        ndarray: matrice N x D del movimento misurato (con errore)
    """
    # Calcola quanto errore aggiungere (deviazione standard)
    sd_mov_e = compute_target_sd(e_imu, ERR_P)
    
    # Genera rumore gaussiano
    mov_error = np.random.randn(np.shape(mov)[0], np.shape(mov)[1])
    
    # Scala rumore alla norma del movimento reale:
    # - mov_error/norm(mov_error) -> direzione casuale
    # - * norm(mov) -> scala alla magnitudine del movimento
    # - * sd_mov_e -> applica fattore di errore relativo
    mov_error = mov_error / np.linalg.norm(mov_error, axis=0) * np.linalg.norm(mov, axis=0) * sd_mov_e
    
    # Movimento misurato = movimento reale + errore
    self_mov = mov + mov_error

    return self_mov
