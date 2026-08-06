"""
LOGICA DECISIONALE
==================

Questo modulo implementa le decisioni di alto livello:
- Quando eseguire MDS (run costoso)
- Quando resettare i nodi (resurface)
- Quando accettare stime MDS vs mantenere stima IMU
- Quale stima scegliere nella retroazione (reverse pass)
"""

import numpy as np
from Positioning.error import estimate_pos_error

# ===== PARAMETRI DECISIONI =====

# RESURFACE: reset periodico della posizione a quella reale
RESURFACE_FREQ = 180    # Resetta ogni N iterazioni
MDS_FREQ = 30           # Esegui MDS ogni N iterazioni

# Not used by Davide
#RESURFACE_TH = 200     
#MDS_TH = 50

# ACCEPTANCE: soglia per accettare stima MDS
ERR_TH = 0.99          # Accetta se errore_nuovo < 0.99 * errore_vecchio


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

def resurface_decision(err, n, ref):
    """
    Decide se fare un reset (resurface) per il nodo ref.
    
    LOGICA:
    - Esegui resurface periodicamente (ogni RESURFACE_FREQ iterazioni)
    - In alternativa: usa soglia di errore (RESURFACE_TH non usato)
    
    Il resurface simula un evento di localizzazione esterna:
    - GPS fix periodico
    - Repositioning manuale
    - Calibrazione periodica
    
    Args:
        err (ndarray): vettore N dell'errore accumulato
        n (int): iterazione attuale
        ref (int): indice del nodo da considerare per resurface
    
    Returns:
        bool: True se va fatto resurface per questo nodo
    """
    # Versione periodica (attiva nel codice)
    return n % RESURFACE_FREQ == 0
    
    # Versione basata su errore (commentata, potrebbe essere migliore)
    # return err[ref] > RESURFACE_TH


def mds_decision(n, err, local, ref=None):
    """
    Decide se eseguire MDS in questa iterazione.
    
    LOGICA:
    - Esegui MDS periodicamente (ogni MDS_FREQ iterazioni)
    - In alternativa: usa soglia di errore medio (non usato)
    
    MDS è computazionalmente costoso, quindi non va fatto ogni iterazione.
    La decisione periodica è semplice, ma potremmo decidere in base
    all'errore accumulato (versioni alternative commentate).
    
    Args:
        n (int): iterazione attuale
        err (ndarray): vettore N dell'errore accumulato
        local (bool): flag per misure locali (non usato nella decisione)
        ref (int): nodo di riferimento (non usato nella decisione)
    
    Returns:
        bool: True se va eseguito MDS
    """
    # Versione periodica (attiva nel codice)
    return n % MDS_FREQ == 0
    
    # Versioni alternative (commentate):
    # return np.mean(err) > MDS_TH        # MDS se errore medio > soglia
    # return max(err) > MDS_TH            # MDS se errore massimo > soglia
    # return err[ref] > MDS_TH            # MDS se errore di ref > soglia


def reverse_decision(new_err, old_err):
    """
    Decide quale stima mantenere nella retroazione backward (reverse pass).
    
    Nel reverse pass, riprocessiamo la sequenza all'indietro per correggere
    le stime usando informazioni "dal futuro".
    
    LOGICA:
    - Se l'errore medio nella retroazione è minore, accetta la retroazione
    - Altrimenti mantieni la stima forward originale
    
    Args:
        new_err (ndarray): vettore N dell'errore DOPO retroazione
        old_err (ndarray): vettore N dell'errore PRIMA retroazione
    
    Returns:
        bool: True se la retroazione ha migliorato (accetta new_err)
    """
    return np.mean(new_err) < np.mean(old_err)
