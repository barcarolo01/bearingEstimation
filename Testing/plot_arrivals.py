import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    h = np.load("H.py")
    
    fs = 96000  # Frequenza di campionamento
    
    # 1. Crea l'asse del tempo (in secondi) basato sul numero di campioni
    tempo = np.arange(len(h)) / fs
    
    mask = (h != 0)
    # 2. Grafica la risposta impulsiva
    plt.figure(figsize=(10, 4))
    plt.stem(tempo[mask], h[mask],linefmt='b-', markerfmt='bo', basefmt='none')
    plt.hlines(y=0, xmin=0, xmax=tempo[mask][-1], color='b', linewidth=1.5)
    
    # 3. Label e dettagli minimi per capire cosa stai guardando
    plt.title("Channel response")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)


    plt.show()