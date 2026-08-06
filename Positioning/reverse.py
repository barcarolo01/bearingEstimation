"""
RETROAZIONE TEMPORALE (REVERSE PASS)
====================================

Questo modulo implementa il "backward pass" della localizzazione:
rielabora l'intera sequenza all'indietro usando tutte le misurazioni
disponibili.

MOTIVATION:
Nella simulazione forward, ogni nodo conosce solo le misurazioni
acquisite fino a quel momento. Nella retroazione, rielaboriamo
tutto sapendo tutte le future misurazioni → possiamo migliorare
le stime di tutti i frame precedenti.

Questo è analogo a:
- Filtraggio offline vs online
- Post-processing di traiettorie registrate
- Smoothing di sequenze temporali
"""

import numpy as np
from Positioning.algo import estimate
from Positioning.decision import reverse_decision
from Positioning.error import estimate_mov_error


def reverse(self_movs, self_errs, self_poss, dists):
    """
    Applica retroazione temporale per migliorare le stime.
    
    ALGORITMO:
    Parte dall'ultimo frame e procede all'indietro:
    
    1. Per ogni iterazione i (dal presente al passato):
       a. Accumula errore dal movimento (norma di mov)
       b. Per ogni nodo n:
          - Se abbiamo misurazione di distanza per questo frame
          - Applica MDS con quella misurazione
       c. Accetta la correzione solo se migliora l'errore
       d. Retrocedi nello spazio (sottrai movimento)
    
    STRUTTURE DATI:
    - self_movs: lista di I matrici N x D (movimento ad ogni iterazione)
    - self_errs: lista di I vettori N (errore ad ogni iterazione)
    - self_poss: lista di I matrici N x D (posizione ad ogni iterazione)
    - dists: lista di N dict dove dists[nodo][iterazione] = dist_matrix
    
    Args:
        self_movs (list): lista di I matrici N x D del movimento
        self_errs (list): lista di I vettori N dell'errore
        self_poss (list): lista di I matrici N x D della posizione
        dists (list): lista di N dict per misurazioni di distanza
    
    Returns:
        tuple: (self_poss aggiornato, self_errs aggiornato)
    """
    
    # ===== INIT STATO RETROAZIONE =====
    # Inizia dall'ultimo frame
    self_mov = self_movs[-1].copy()
    self_err_rev = self_errs[-1].copy()
    self_pos_rev = self_poss[-1].copy()

    # Numero di frame
    pos_l = len(self_poss)
    
    # ===== LOOP RETROAZIONE (all'indietro) =====
    for i in range(pos_l - 1, 0 - 1, -1):
        # ----- ACCUMULA ERRORE DAL MOVIMENTO -----
        # Nel retroazionare, il movimento aggiunge ancora errore
        self_err_rev += estimate_mov_error(self_mov)

        # ----- COPIA STATO PRIMA DI TENTARE MDS -----
        self_pos_rev_copy = self_pos_rev.copy()
        self_err_rev_copy = self_err_rev.copy()
        
        # ----- MDS PER OGNI NODO (se disponibile misurazione) -----
        # Nota: nel reverse pass, ogni nodo processa le sue misurazioni
        for n in range(len(self_pos_rev)):
            # Controlla se c'è una misurazione di distanza per questo nodo
            # al frame i
            if i in dists[n].keys():
                # Estrai la matrice di distanze misurata
                measured_dist = dists[n][i]

                # Applica MDS per stimare nuove posizioni
                # (usa l'intera matrice di distanze, non solo per nodo n)
                self_pos_rev_new, self_err_rev_new = estimate(
                    self_pos_rev_copy,
                    self_err_rev_copy,
                    measured_dist
                )

                # Accetta la nuova stima per il nodo n
                # (gli altri nodi vengono rielaborati dai loro MDS)
                self_pos_rev[n, :] = self_pos_rev_new[n, :].copy()
                self_err_rev[n] = self_err_rev_new[n]

        # ----- DECIDE SE ACCETTARE RETROAZIONE -----
        # Confronta errore dopo retroazione vs errore del forward pass
        if reverse_decision(self_err_rev, self_errs[i]):
            # Retroazione ha migliorato: accetta
            self_poss[i] = self_pos_rev.copy()
            self_errs[i] = self_err_rev.copy()
            print(f"  Frame {i}: retroazione accettata")
        else:
            # Retroazione ha peggiorato: mantieni forward pass
            self_pos_rev = self_poss[i].copy()
            self_err_rev = self_errs[i]
            print(f"  Frame {i}: retroazione rifiutata")

        # ----- RETROCEDI NELLO SPAZIO =====
        # Applica movimento in senso inverso per il frame precedente
        # Nota: sottraiamo il movimento perché stiamo andando indietro
        self_mov = self_movs[i].copy()
        self_pos_rev -= self_mov

    return self_poss, self_errs


# ===== SPIEGAZIONE DETTAGLIATA DELL'ALGORITMO =====
"""
ESEMPIO CON 3 FRAME (I=2, i=0,1):

Forward pass (sim.py):
  Frame 0: pos_0, err_0 (init)
  Frame 1: pos_1, err_1 (IMU + possibile MDS)
  Frame 2: pos_2, err_2 (IMU + possibile MDS)

Reverse pass (questo file):
  Parte da i=1 (penultimo frame) all'indietro fino a i=0

  Iterazione 1 (rielabora frame 1):
    - Leggi: self_movs[1] (movimento dal frame 1 al 2)
    - Accumula errore
    - Applica MDS se c'è misurazione per frame 1
    - Decide se accettare
    - Sottrai movimento: pos_rev -= self_movs[1]
      → Ora pos_rev è allineato al frame 0

  Iterazione 2 (rielabora frame 0):
    - Leggi: self_movs[0] (movimento dal frame 0 al 1)
    - Accumula errore
    - Applica MDS se c'è misurazione per frame 0
    - Decide se accettare
    - Sottrai movimento: pos_rev -= self_movs[0]
      → Ora pos_rev è allineato al frame -1 (non usato)

EFFETTO:
- Frame 1 viene corretto sapendo che frame 2 ha una misurazione
- Frame 0 viene corretto sapendo che frame 1 ha una misurazione
- Questo è il "smoothing" temporale

NOTA IMPORTANTE:
Il reverse pass NON rielabora il frame I (ultimo), solo i frame
precedenti. Sarebbe utile aggiungere un ulteriore backward pass
per correggere anche l'ultimo frame.
"""
