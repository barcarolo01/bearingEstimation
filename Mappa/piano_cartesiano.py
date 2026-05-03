import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
'''
bearing_H1  = [315,269 ,225]   # gradi da A — modifica qui
bearing_H2 = [45, 91, 135]
'''

bearing_H1  = np.load("H1.npy")
bearing_H2 = np.load("H2.npy")


def disegna_semiretta(ax, origine, alpha_gradi, lunghezza=8, colore='green', etichetta=None):
    """
    Disegna una semiretta su un piano cartesiano.

    Parametri:
        ax          : assi matplotlib su cui disegnare
        origine     : tupla (x, y) - punto di partenza della semiretta
        alpha_gradi : angolo in gradi rispetto all'asse X positivo, misurato in senso antiorario
        lunghezza   : lunghezza visiva della semiretta (default 8)
        colore      : colore della semiretta (default 'green')
        etichetta   : testo opzionale da mostrare vicino alla semiretta
    """
    alpha_rad = np.radians(alpha_gradi)
    dx = lunghezza * np.cos(alpha_rad)
    dy = lunghezza * np.sin(alpha_rad)
    x0, y0 = origine

    ax.plot([x0, x0 + dx], [y0, y0 + dy], color=colore, linewidth=2, zorder=4)

    ax.quiver(
        x0, y0, dx, dy,
        angles='xy', scale_units='xy', scale=1,
        color=colore, width=0.005, headwidth=4, headlength=5,
        zorder=5
    )

    if etichetta:
        offset = 0.6
        perp_dx = -np.sin(alpha_rad) * offset
        perp_dy =  np.cos(alpha_rad) * offset
        ax.text(
            x0 + dx * 0.55 + perp_dx,
            y0 + dy * 0.55 + perp_dy,
            etichetta,
            fontsize=11, color=colore, fontweight='bold',
            ha='center', va='center'
        )


def calcola_intersezione(origine1, alpha_gradi1, origine2, alpha_gradi2):
    """
    Calcola il punto di intersezione tra due semirette.

    Ogni semiretta è definita da:
        P(t) = origine + t * direzione,  con t >= 0

    Risolve il sistema lineare:
        origine1 + t1 * dir1 = origine2 + t2 * dir2

    Restituisce:
        (x, y)  se le semirette si intersecano (t1 >= 0 e t2 >= 0)
        None    se sono parallele o l'intersezione è sul prolungamento
    """
    x1, y1 = origine1
    x2, y2 = origine2

    a1_rad = np.radians(alpha_gradi1)
    a2_rad = np.radians(alpha_gradi2)

    # Direzioni unitarie
    d1x, d1y = np.cos(a1_rad), np.sin(a1_rad)
    d2x, d2y = np.cos(a2_rad), np.sin(a2_rad)

    # Sistema: x1 + t1*d1x = x2 + t2*d2x
    #          y1 + t1*d1y = y2 + t2*d2y
    # In forma matriciale: A * [t1, t2]^T = b
    A = np.array([[d1x, -d2x],
                  [d1y, -d2y]])
    b = np.array([x2 - x1, y2 - y1])

    det = np.linalg.det(A)
    if abs(det) < 1e-10:
        # Rette parallele o coincidenti
        return None

    t1, t2 = np.linalg.solve(A, b)

    if t1 < -1e-9 or t2 < -1e-9:
        # L'intersezione esiste sulla retta ma non sulle semirette
        return None

    xi = x1 + t1 * d1x
    yi = y1 + t1 * d1y
    return (xi, yi)


# ── Configurazione ────────────────────────────────────────────────────────────

PUNTO_A  = (0, 10)
ALPHA_A  = 269   # gradi da A — modifica qui

PUNTO_B  = (0, -10)
ALPHA_B  = 91    # gradi da B — modifica qui

LUNG_SEMIRETTA = 8  # lunghezza visiva delle semirette

# ── Figura ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 8))

ax.axhline(0, color='black', linewidth=1.2)
ax.axvline(0, color='black', linewidth=1.2)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)

# Punti A e B
for nome, (x, y), col in [('A', PUNTO_A, 'royalblue'), ('B', PUNTO_B, 'tomato')]:
    ax.plot(x, y, 'o', color=col, markersize=12, zorder=6)
    ax.annotate(f'{nome} ({x}, {y})', xy=(x, y), xytext=(0.5, y),
                fontsize=12, fontweight='bold', color=col, va='center')
 
# Itera sulle coppie di angoli
trovate = 0
for i, (alpha_a, alpha_b) in enumerate(zip(bearing_H1,bearing_H2)):
 
 
    intersezione = calcola_intersezione(PUNTO_A, alpha_a, PUNTO_B, alpha_b)
 
    if intersezione:
        ix, iy = intersezione
        ax.plot(ix, iy, '.', color='gold', markersize=16, zorder=7,
                markeredgecolor='darkorange', markeredgewidth=1.2)
 
        trovate += 1
        print(f"Coppia {i+1} (α_A={alpha_a}°, α_B={alpha_b}°): intersezione in ({ix:.2f}, {iy:.2f})")
    else:
        print(f"Coppia {i+1} (α_A={alpha_a}°, α_B={alpha_b}°): nessuna intersezione")

 
# ── Stile ─────────────────────────────────────────────────────────────────────
 
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.set_xlabel('X', fontsize=13, labelpad=10)
ax.set_ylabel('Y', fontsize=13, labelpad=10)
ax.set_title('Piano Cartesiano — Intersezioni tra coppie di semirette',
             fontsize=12, fontweight='bold', pad=15)
 
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.plot(1, 0, '>k', transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(0, 1, '^k', transform=ax.get_xaxis_transform(), clip_on=False)
 
plt.tight_layout()
plt.show()
 