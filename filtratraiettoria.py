import numpy as np
from scipy.ndimage import median_filter as nd_medfilt

def remove_outliers_median(coordinate, dimensione_finestra=11):
    dati_puliti = np.zeros_like(coordinate)
    
    for i in range(coordinate.shape[1]):
        # 'reflect' specchia i dati sul bordo, 'nearest' ripete l'ultimo valore.
        # Entrambi evitano l'effetto distorsivo degli zeri alla fine della traiettoria.
        dati_puliti[:, i] = nd_medfilt(coordinate[:, i], size=dimensione_finestra, mode='nearest')
        
    return dati_puliti

def group_close_points(dati, tolleranza_metri=2.0):
    """
    Raggruppa i punti consecutivi troppo vicini tra loro, 
    sostituendoli con la media geometrica del gruppo.
    """
    if len(dati) == 0:
        return dati
        
    METRI_PER_GRADO_LAT = 111320.0
    METRI_PER_GRADO_LON = 104600.0
    
    punti_filtrati = []
    cluster_corrente = [dati[0]]
    
    for i in range(1, len(dati)):
        # Calcola distanza tra il punto corrente e l'inizio del cluster
        diff_lat = (dati[i, 0] - cluster_corrente[0][0]) * METRI_PER_GRADO_LAT
        diff_lon = (dati[i, 1] - cluster_corrente[0][1]) * METRI_PER_GRADO_LON
        distanza = np.sqrt(diff_lat**2 + diff_lon**2)
        
        if distanza <= tolleranza_metri:
            # Il punto è molto vicino, lo aggiungo al gruppo per fare la media
            cluster_corrente.append(dati[i])
        else:
            # Il punto si è allontanato: chiudo il cluster precedente facendo la media
            punti_filtrati.append(np.mean(cluster_corrente, axis=0))
            # Faccio partire un nuovo cluster con il punto attuale
            cluster_corrente = [dati[i]]
            
    # Non dimentichiamo l'ultimo cluster rimasto aperto
    punti_filtrati.append(np.mean(cluster_corrente, axis=0))
    
    return np.array(punti_filtrati)


def compute_geometric_rmse(ground_truth, stime):
    """
    Calcola l'errore RMS (in metri) tra Ground Truth equispaziata e punti stimati
    trovando la minima distanza geometrica per ogni punto stimato.
    
    :param ground_truth: Array numpy (M, 2) [Lat, Lon] della traiettoria reale
    :param stime: Array numpy (N, 2) [Lat, Lon] dei punti stimati (filtrati o grezzi)
    :return: Valore float dell'RMSE espresso in metri
    """
    # Fattori di conversione locali in metri (assumendo ~20°N)
    METRI_PER_GRADO_LAT = 111320.0
    METRI_PER_GRADO_LON = 104600.0
    
    # 1. Convertiamo l'intera Ground Truth in coordinate metriche locali (es. relative al primo punto)
    # Questo serve per calcolare le distanze euclidee reali in metri
    gt_lat_m = ground_truth[:, 0] * METRI_PER_GRADO_LAT
    gt_lon_m = ground_truth[:, 1] * METRI_PER_GRADO_LON
    
    stime_lat_m = stime[:, 0] * METRI_PER_GRADO_LAT
    stime_lon_m = stime[:, 1] * METRI_PER_GRADO_LON
    
    min_distanze_metri = []
    
    # 2. Per ogni punto stimato, trova la minima distanza dal perimetro della GT
    for i in range(len(stime)):
        # Distanza tra il punto stimato i-esimo e TUTTI i punti della GT
        diff_lat = gt_lat_m - stime_lat_m[i]
        diff_lon = gt_lon_m - stime_lon_m[i]
        distanze_all_gt = np.sqrt(diff_lat**2 + diff_lon**2)
        
        # Prendiamo solo la distanza dal punto della GT più vicino
        min_distanze_metri.append(np.min(distanze_all_gt))
        
    min_distanze_metri = np.array(min_distanze_metri)
    
    # 3. Calcolo del Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean(min_distanze_metri**2))
    
    return rmse