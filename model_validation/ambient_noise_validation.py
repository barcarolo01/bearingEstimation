import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from scipy.signal import welch
from digitalshadow.underwater_sim.ambient_noise import add_ambient_noise, generate_ambient_noise, P_REF

FONTSIZE = 15

# Per arrivare a 1e6 Hz come nella Fig. 2 di Stojanovic serve FS/2 >= 1e6 (Nyquist).
FS = 2_000_000
DURATION_S = 12
NPERSEG = 2 ** 22                      # df = FS/NPERSEG ~ 0.48 Hz -> risolve bene fino a ~1 Hz
PTS_PER_DECADE = 60                    # punti per decade nel plot (media in potenza dei bin di Welch)
F_MIN, F_MAX = 1.0, 1e6                # stesso range della figura (10^0 - 10^6 Hz)
Y_MIN, Y_MAX = 20, 110                 # stesso range verticale della figura
F_LOW, F_HIGH = 500.0, 5_000.0         # banda per il livello integrato

# Shipping activity: solo i due estremi. (valore, etichetta)
# L'ordine decide anche il tono: il primo usa il colore scuro, il secondo quello chiaro.
SHIPPING_LEVELS = [
    (0.0, "Shipping 0 (low)"),
    (1.0, "Shipping 1 (high)"),
]
LEGEND_GREY = ("#3b3b39", "#b3b2ac")   # swatch grigi della legenda: scuro / chiaro

# Vento: (etichetta, m/s, (colore shipping 0 = scuro, colore shipping 1 = chiaro))
# Una tinta per vento; la luminosita' codifica lo shipping.
CONFIGS = [
    ("0 m/s - calm sea",      0.0, ("#1b529f", "#7ec0ff")),   # blu
    ("5 m/s - light breeze",  5.0, ("#c60300", "#c96666")),   # arancio
    ("10 m/s - strong wind", 10.0, ("#0d813a", "#5fe08c")),   # verde
    ("15 m/s - rough sea",   15.0, ("#c9049e", "#f784da")),   # rosso
]

MODEL_STYLE = dict(color="black", lw=0.9, ls=(0, (4, 3)), alpha=0.8)

plt.rcParams.update({
    "font.size":        FONTSIZE,
    "axes.titlesize":   FONTSIZE,
    "axes.labelsize":   FONTSIZE,
    "xtick.labelsize":  FONTSIZE,
    "ytick.labelsize":  FONTSIZE,
    "legend.fontsize":  FONTSIZE,
    "figure.titlesize": FONTSIZE,
})


def band_level_db(f, psd, f_low, f_high):
    m = (f >= f_low) & (f <= f_high)
    return 10 * np.log10(np.trapezoid(psd[m], f[m]) / P_REF ** 2)


def log_binned_db(f, pxx, f_edges):
    """Media in potenza dei bin di Welch dentro ogni intervallo logaritmico -> PSD [dB re 1 µPa²/Hz]."""
    idx = np.digitize(f, f_edges)
    out = np.full(len(f_edges) - 1, np.nan)
    for k in range(1, len(f_edges)):
        m = idx == k
        if m.any():
            out[k - 1] = pxx[m].mean()
    centres = np.sqrt(f_edges[:-1] * f_edges[1:])
    ok = ~np.isnan(out)
    return centres[ok], 10 * np.log10(out[ok] / P_REF ** 2)


def build_legend(ax):
    """Legenda: colore = vento (mostra entrambi i toni), tono = shipping, nero = modello."""
    handles, labels = [], []
    for label, _, colours in CONFIGS:
        handles.append(tuple(Line2D([], [], color=c, lw=2.2) for c in colours))
        labels.append(label)
    for grey, (_, ship_lbl) in zip(LEGEND_GREY, SHIPPING_LEVELS):
        handles.append(Line2D([], [], color=grey, lw=2.2))
        #labels.append(ship_lbl)
    handles.append(Line2D([], [], **MODEL_STYLE))
    labels.append("Model (analytic)")
    ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=FONTSIZE,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})


def main():
    rng = np.random.default_rng(0)
    silence = np.zeros(FS * DURATION_S)
    f_model = np.logspace(np.log10(F_MIN), np.log10(F_MAX), 800)
    n_dec = np.log10(F_MAX / F_MIN)
    f_edges = np.logspace(np.log10(F_MIN), np.log10(F_MAX), int(n_dec * PTS_PER_DECADE) + 1)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    rows = []
    for label, wind, colours in CONFIGS:
        for tone, ((ship, ship_lbl), colour) in enumerate(zip(SHIPPING_LEVELS, colours)):
            noise, _ = add_ambient_noise([silence], FS, shipping=ship, wind=wind,
                                         rng=rng, verbose=False)
            f, pxx = welch(noise[0], FS, nperseg=NPERSEG)
            del noise
            rows.append((f"{label}, {ship_lbl}", band_level_db(f, pxx, F_LOW, F_HIGH)))

            # PSD generata (colore, linea solida), mediata su bin logaritmici
            # (esclude DC e il bin vicino a Nyquist)
            keep = (f >= F_MIN) & (f < F_MAX)
            fc, pxx_db = log_binned_db(f[keep], pxx[keep], f_edges)
            ax.semilogx(fc, pxx_db, color=colour, lw=2.2, ls="-",
                        zorder=3 - 0.1 * tone,       # tono scuro sopra quello chiaro
                        solid_capstyle="round")

            # modello analitico (nero), uno per ciascun caso (vento x shipping)
            ax.semilogx(f_model, generate_ambient_noise(f_model, ship, wind),
                        zorder=4, **MODEL_STYLE)

    build_legend(ax)

    ax.set_xlabel("Frequency [Hz]", fontsize=FONTSIZE, color="#52514e")
    ax.set_ylabel("PSD [dB re 1 µPa²/Hz]", fontsize=FONTSIZE, color="#52514e")
    ax.set_xlim(F_MIN, F_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.grid(True, which="major", lw=0.6, zorder=1, ls="--")
    ax.grid(True, which="minor", lw=0.35, alpha=0.6, zorder=1, ls="--")
    ax.tick_params(colors="#52514e", labelsize=FONTSIZE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.tight_layout()
    fig.savefig("ambient_noise_psd.pdf")

    print(f"Band level over {F_LOW:.0f}-{F_HIGH:.0f} Hz:")
    for label, lvl in rows:
        print(f"  {label:<40s} {lvl:6.1f} dB re 1 uPa")


if __name__ == "__main__":
    main()