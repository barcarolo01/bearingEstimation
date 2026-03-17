"""
============================================================
  STIMA DELLA DIREZIONE DI ARRIVO (DOA) - ARRAY 3 IDROFONI
============================================================

Array: triangolo equilatero con lato 100 cm
Metodo: GCC-PHAT per stimare i ritardi temporali (TDOA)
        + soluzione geometrica per il calcolo dell'azimut

Come usarlo:
    1. Installa le dipendenze (una volta sola):
           pip install numpy scipy soundfile matplotlib

    2. Modifica i parametri nella sezione "PARAMETRI CONFIGURABILI"

    3. Esegui:
           python doa_motoscafo.py
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ============================================================
#  PARAMETRI CONFIGURABILI  <-- modifica questi valori
# ============================================================

PERCORSO_FILE_WAV = "0958_.wav"   # percorso del tuo file .wav

LATO_TRIANGOLO_M  = 1.00    # lato del triangolo equilatero in metri
VELOCITA_SUONO    = 1500.0  # velocita del suono in acqua [m/s]

# Filtro passa-banda per isolare il rumore del motoscafo
FREQ_MIN_HZ = 50.0          # frequenza minima [Hz]
FREQ_MAX_HZ = 10000.0        # frequenza massima [Hz]

# Rilevamento del motoscafo: finestra e soglia energia
FINESTRA_SEC      = 1     # durata finestra per il calcolo dell'energia RMS [secondi]
SOGLIA_ENERGIA    = 1.0     # fattore moltiplicativo rispetto all'energia media
                            # (aumenta se rileva troppi falsi positivi,
                            #  diminuisci se non trova segmenti)
DURATA_MINIMA_SEC = 5.0     # scarta segmenti piu' corti di questa durata [secondi]
                            # utile per eliminare spike/falsi positivi brevi

# Stima DOA: sotto-finestre all'interno di ogni segmento
FINESTRA_DOA_SEC  = 0.1     # durata di ogni sotto-finestra per la stima DOA [secondi]
                            # deve essere <= DURATA_MINIMA_SEC
                            # finestre piu' corte = piu' stime, ma meno accurate
OVERLAP_DOA       = 0.1     # sovrapposizione tra sotto-finestre consecutive [0.0 - 0.9]
                            # 0.5 = 50% di overlap, produce piu' punti nel grafico

# ============================================================
#  GEOMETRIA DELL'ARRAY
# ============================================================
# Triangolo equilatero: idrofono 0 in alto, 1 e 2 in basso a sx e dx
#
#         H0  (0, h)
#        /    \
#      H1 ---- H2
#  (-d, 0)   (+d, 0)
#
# con d = lato/2  e  h = lato * sqrt(3)/2

def calcola_posizioni_idrofoni(lato):
    """Restituisce le coordinate (x, y) dei 3 idrofoni in metri."""
    d = lato / 2.0
    h = lato * np.sqrt(3) / 2.0
    posizioni = np.array([
        [ 0.0,  h],    # H0 - in alto
        [-d,    0.0],  # H1 - in basso a sinistra
        [ d,    0.0]   # H2 - in basso a destra
    ])
    return posizioni

# ============================================================
#  FUNZIONI DI ELABORAZIONE
# ============================================================

def carica_wav(percorso):
    """Legge il file WAV e restituisce i primi 3 canali."""
    print(f"Caricamento file: {percorso}")
    dati, fs = sf.read(percorso)

    if dati.ndim == 1:
        raise ValueError("Il file WAV ha un solo canale. Attesi almeno 3 canali.")
    if dati.shape[1] < 3:
        raise ValueError(f"Il file ha solo {dati.shape[1]} canali. Attesi almeno 3.")

    h0 = dati[:, 0]
    h1 = dati[:, 1]
    h2 = dati[:, 2]

    durata = len(h0) / fs
    print(f"  Frequenza di campionamento : {fs} Hz")
    print(f"  Durata registrazione       : {durata:.1f} secondi")
    print(f"  Campioni totali            : {len(h0)}")
    return h0, h1, h2, fs


def filtra_banda(segnale, fs, f_min, f_max):
    """Applica un filtro Butterworth passa-banda."""
    nyq = fs / 2.0
    low  = f_min / nyq
    high = f_max / nyq
    low  = max(low,  1e-4)
    high = min(high, 0.9999)
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, segnale)


def rileva_segmenti_motoscafo(segnale, fs, finestra_sec, soglia_fattore, durata_minima_sec):
    """
    Individua gli intervalli temporali in cui e' presente il motoscafo,
    basandosi sull'energia RMS su finestre scorrevoli.

    Parametri:
        durata_minima_sec : segmenti piu' brevi di questo valore vengono scartati
                            (elimina spike e falsi positivi)

    Restituisce:
        segmenti_validi : lista di tuple (t_inizio, t_fine) in secondi
        energie         : array delle energie RMS per ogni finestra
        soglia          : valore di soglia utilizzato
    """
    n_camp_finestra = int(finestra_sec * fs)
    n_finestre = len(segnale) // n_camp_finestra

    energie = []
    for i in range(n_finestre):
        segmento = segnale[i * n_camp_finestra : (i + 1) * n_camp_finestra]
        rms = np.sqrt(np.mean(segmento ** 2))
        energie.append(rms)

    energie = np.array(energie)
    soglia  = np.mean(energie) * soglia_fattore

    # Trova le finestre sopra soglia e raggruppa in segmenti continui
    attivo = energie > soglia
    segmenti = []
    in_segmento = False
    t_inizio = 0.0

    for i, a in enumerate(attivo):
        t = i * finestra_sec
        if a and not in_segmento:
            t_inizio = t
            in_segmento = True
        elif not a and in_segmento:
            segmenti.append([t_inizio, t])
            in_segmento = False

    if in_segmento:
        segmenti.append([t_inizio, n_finestre * finestra_sec])

    # Unisci segmenti vicini (distanza < 1 secondo)
    segmenti_uniti = []
    for seg in segmenti:
        if segmenti_uniti and seg[0] - segmenti_uniti[-1][1] < 1.0:
            segmenti_uniti[-1][1] = seg[1]
        else:
            segmenti_uniti.append(seg)

    # Scarta segmenti troppo brevi (spike / falsi positivi)
    segmenti_validi = [(s[0], s[1]) for s in segmenti_uniti if s[1] - s[0] >= durata_minima_sec]
    n_scartati = len(segmenti_uniti) - len(segmenti_validi)
    if n_scartati > 0:
        print(f"  Segmenti scartati (durata < {durata_minima_sec}s): {n_scartati}")

    return segmenti_validi, energie, soglia


def gcc_phat(x, y, fs):
    """
    Stima il ritardo temporale tra due segnali con il metodo GCC-PHAT.
    Restituisce il ritardo in secondi (positivo = x arriva prima di y).
    """
    n = len(x) + len(y) - 1
    X = np.fft.rfft(x, n=n)
    Y = np.fft.rfft(y, n=n)
    R = X * np.conj(Y)
    denom = np.abs(R)
    denom[denom < 1e-10] = 1e-10
    R_phat = R / denom
    cc = np.fft.irfft(R_phat, n=n)
    lags = np.arange(-(len(x) - 1), len(y))
    idx_picco = np.argmax(np.abs(cc))
    ritardo_campioni = lags[idx_picco]
    return ritardo_campioni / fs


def stima_doa(tau_01, tau_02, tau_12, posizioni, c):
    """
    Stima l'angolo di arrivo (azimut in gradi) tramite ricerca su griglia.
    Convenzione: 0 deg = direzione verso H0, angoli in senso antiorario.
    """
    angoli_test = np.linspace(0, 360, 3600)  # risoluzione 0.1 gradi
    errore_minimo = np.inf
    angolo_migliore = 0.0

    p0, p1, p2 = posizioni[0], posizioni[1], posizioni[2]

    for theta_deg in angoli_test:
        theta = np.radians(theta_deg)
        direzione = np.array([np.sin(theta), np.cos(theta)])

        tau_01_pred = np.dot(p0 - p1, direzione) / c
        tau_02_pred = np.dot(p0 - p2, direzione) / c
        tau_12_pred = np.dot(p1 - p2, direzione) / c

        errore = ((tau_01 - tau_01_pred) ** 2 +
                  (tau_02 - tau_02_pred) ** 2 +
                  (tau_12 - tau_12_pred) ** 2)

        if errore < errore_minimo:
            errore_minimo = errore
            angolo_migliore = theta_deg

    return angolo_migliore


def stima_doa_segmento(h0f, h1f, h2f, fs, t0, t1,
                        finestra_doa_sec, overlap, posizioni, c):
    """
    Divide il segmento [t0, t1] in sotto-finestre con overlap e stima
    la DOA per ognuna. Restituisce una lista di dict con t_centro e azimut.

    Parametri:
        finestra_doa_sec : durata di ogni sotto-finestra [secondi]
        overlap          : frazione di sovrapposizione tra finestre [0.0 - 0.9]
    """
    tau_max  = LATO_TRIANGOLO_M / VELOCITA_SUONO
    passo    = finestra_doa_sec * (1.0 - overlap)   # avanzamento tra finestre
    n_finestra = int(finestra_doa_sec * fs)
    n_passo    = max(1, int(passo * fs))

    c_inizio = int(t0 * fs)
    c_fine   = int(t1 * fs)

    stime = []
    pos = c_inizio

    while pos + n_finestra <= c_fine:
        s0 = h0f[pos : pos + n_finestra]
        s1 = h1f[pos : pos + n_finestra]
        s2 = h2f[pos : pos + n_finestra]

        tau_01 = np.clip(gcc_phat(s0, s1, fs), -tau_max, tau_max)
        tau_02 = np.clip(gcc_phat(s0, s2, fs), -tau_max, tau_max)
        tau_12 = np.clip(gcc_phat(s1, s2, fs), -tau_max, tau_max)

        azimut = stima_doa(tau_01, tau_02, tau_12, posizioni, c)

        t_centro = (pos + n_finestra / 2) / fs
        stime.append({
            "t_centro"  : t_centro,
            "azimut_deg": azimut,
            "tau_01_us" : tau_01 * 1e6,
            "tau_02_us" : tau_02 * 1e6,
            "tau_12_us" : tau_12 * 1e6,
        })

        pos += n_passo

    return stime


# ============================================================
#  PROGRAMMA PRINCIPALE
# ============================================================

def main():

    # 1. Carica il file WAV
    h0, h1, h2, fs = carica_wav(PERCORSO_FILE_WAV)

    # 2. Filtra nella banda di interesse
    print(f"\nFiltraggio banda {FREQ_MIN_HZ:.0f}-{FREQ_MAX_HZ:.0f} Hz...")
    h0f = filtra_banda(h0, fs, FREQ_MIN_HZ, FREQ_MAX_HZ)
    h1f = filtra_banda(h1, fs, FREQ_MIN_HZ, FREQ_MAX_HZ)
    h2f = filtra_banda(h2, fs, FREQ_MIN_HZ, FREQ_MAX_HZ)

    # 3. Rileva i segmenti con il motoscafo
    print(f"\nRilevamento segmenti (finestra={FINESTRA_SEC}s, "
          f"soglia x{SOGLIA_ENERGIA}, durata minima={DURATA_MINIMA_SEC}s)...")
    segmenti, energie, soglia = rileva_segmenti_motoscafo(
        h0f, fs, FINESTRA_SEC, SOGLIA_ENERGIA, DURATA_MINIMA_SEC
    )
    print(f"  Segmenti validi trovati: {len(segmenti)}")
    for i, (t0, t1) in enumerate(segmenti):
        print(f"    Segmento {i+1}: {t0:.1f}s -> {t1:.1f}s  (durata {t1-t0:.1f}s)")

    if not segmenti:
        print("\nNessun segmento rilevato. Prova ad abbassare SOGLIA_ENERGIA o DURATA_MINIMA_SEC.")
        return

    # 4. Geometria array
    posizioni = calcola_posizioni_idrofoni(LATO_TRIANGOLO_M)

    # 5. Stima DOA con sotto-finestre per ogni segmento
    print(f"\nStima DOA (finestra={FINESTRA_DOA_SEC}s, overlap={int(OVERLAP_DOA*100)}%)...")
    risultati_per_segmento = []   # lista di liste di stime

    for i, (t0, t1) in enumerate(segmenti):
        durata = t1 - t0
        stime = stima_doa_segmento(
            h0f, h1f, h2f, fs, t0, t1,
            FINESTRA_DOA_SEC, OVERLAP_DOA, posizioni, VELOCITA_SUONO
        )
        risultati_per_segmento.append(stime)
        print(f"  Segmento {i+1} ({t0:.1f}s-{t1:.1f}s, durata {durata:.1f}s): "
              f"{len(stime)} stime DOA")

        for s in stime:
            print(f"    t={s['t_centro']:.2f}s  azimut={s['azimut_deg']:.1f} deg  "
                  f"[tau01={s['tau_01_us']:.1f}us, "
                  f"tau02={s['tau_02_us']:.1f}us, "
                  f"tau12={s['tau_12_us']:.1f}us]")

    # 6. Plot
    fig, assi = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle("Analisi DOA - Rumore Motoscafo", fontsize=14, fontweight='bold')

    # --- Plot energia nel tempo ---
    ax1 = assi[0]
    t_energia = np.arange(len(energie)) * FINESTRA_SEC
    ax1.plot(t_energia, energie, color='steelblue', label='Energia RMS (H0)')
    ax1.axhline(soglia, color='red', linestyle='--', label=f'Soglia (x{SOGLIA_ENERGIA})')
    for i, (t0, t1) in enumerate(segmenti):
        ax1.axvspan(t0, t1, alpha=0.2, color='orange',
                    label='Motoscafo rilevato' if i == 0 else "")
    ax1.set_xlabel("Tempo [s]")
    ax1.set_ylabel("Energia RMS")
    ax1.set_title("Energia del segnale e segmenti rilevati")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot azimut nel tempo (tutte le stime) ---
    ax2 = assi[1]

    # Colori diversi per ogni segmento
    colori = plt.cm.tab10(np.linspace(0, 1, max(len(segmenti), 1)))

    for i, (stime, (t0, t1)) in enumerate(zip(risultati_per_segmento, segmenti)):
        if not stime:
            continue
        t_vals = [s["t_centro"]   for s in stime]
        a_vals = [s["azimut_deg"] for s in stime]
        colore = colori[i % len(colori)]

        # Sfondo del segmento
        ax2.axvspan(t0, t1, alpha=0.08, color=colore)
        # Linea che unisce le stime
        ax2.plot(t_vals, a_vals, color=colore, linewidth=1.5,
                 linestyle='--', alpha=0.6)
        # Punti delle stime
        ax2.scatter(t_vals, a_vals, color=colore, s=60, zorder=5,
                    label=f"Segmento {i+1}")
        # Etichette angolo
        for t, a in zip(t_vals, a_vals):
            ax2.annotate(f"{a:.0f}°",
                         xy=(t, a), xytext=(4, 4),
                         textcoords='offset points', fontsize=8, color=colore)

    ax2.set_xlabel("Tempo [s]")
    ax2.set_ylabel("Azimut stimato [gradi]")
    ax2.set_title(f"Direzione di arrivo nel tempo "
                  f"(finestra DOA={FINESTRA_DOA_SEC}s, overlap={int(OVERLAP_DOA*100)}%)")
    ax2.set_ylim(0, 360)
    ax2.set_yticks(range(0, 361, 45))

    # Etichette cardinali sull'asse Y (basate sulla convenzione dell'array)
    etichette_y = {0: "0°\n(→H0)", 45: "45°", 90: "90°", 135: "135°",
                   180: "180°", 225: "225°", 270: "270°", 315: "315°", 360: "360°"}
    ax2.set_yticklabels([etichette_y.get(v, f"{v}°") for v in range(0, 361, 45)])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("doa_risultati.png", dpi=150, bbox_inches='tight')
    print("\nPlot salvato in: doa_risultati.png")
    plt.show()

    # 7. Riepilogo finale
    print("\n" + "="*60)
    print("RIEPILOGO RISULTATI")
    print("="*60)
    angolomedio = 0
    numeroangoli = 0
    for i, (stime, (t0, t1)) in enumerate(zip(risultati_per_segmento, segmenti)):
        print(f"\nSegmento {i+1}  ({t0:.1f}s - {t1:.1f}s, durata {t1-t0:.1f}s)")
        print(f"  {'T centro':>10}   {'Azimut':>8}")
        print(f"  {'-'*22}")
        for s in stime:
            print(f"  {s['t_centro']:>8.2f}s   {s['azimut_deg']:>6.1f} deg")
            angolomedio+=s['azimut_deg']
            numeroangoli += 1
        angolomedio = angolomedio / numeroangoli
        print(f"ANGOLO MEDIO = {angolomedio}")
    print("\n" + "="*60)
    print("Convenzione angolare:")
    print("  0°   = direzione verso H0 (idrofono in alto)")
    print("  90°  = direzione verso destra")
    print("  180° = direzione opposta a H0")
    print("  270° = direzione verso sinistra")


if __name__ == "__main__":
    main()
