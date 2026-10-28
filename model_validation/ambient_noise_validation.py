import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from digitalshadow.underwater_sim.ambient_noise import add_ambient_noise, wenz_psd_db, P_REF

FONTSIZE = 15

FS = 96_000
DURATION_S = 20
F_LOW, F_HIGH = 500.0, 5_000.0          # analysis band
SHIPPING = 0.5                        # fixed: it only matters below ~1 kHz

CONFIGS = [
    ("0 m/s - calm sea",      0.0, "#2a78d6"),
    ("5 m/s - light breeze",  5.0, "#eb6834"),
    ("10 m/s - strong wind", 10.0, "#1baf7a"),
    ("15 m/s - rough sea",   15.0, "#e34948"),
]

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

def main():
    rng = np.random.default_rng(0)
    silence = np.zeros(FS * DURATION_S)
    f_model = np.logspace(np.log10(90), np.log10(FS / 2), 600)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    rows = []
    for label, wind, colour in CONFIGS:
        noise,_ = add_ambient_noise([silence], FS, shipping=SHIPPING, wind=wind,
                                  rng=rng, verbose=False)
                            
        f, pxx = welch(noise[0], FS, nperseg=8192)
        keep = f > 0
        keep[-1] = False
        ax.semilogx(f[keep], 10 * np.log10(pxx[keep] / P_REF ** 2),
                    color=colour, lw=2.0, zorder=3, solid_capstyle="round")
        ax.semilogx(f_model, wenz_psd_db(f_model, SHIPPING, wind),
                    color="#52514e", lw=1.0, ls=(0, (4, 3)), zorder=4, alpha=0.75)
        rows.append((label, band_level_db(f, pxx, F_LOW, F_HIGH)))

        ax.annotate(label,
                    xy=(FS / 2 * 0.95, wenz_psd_db(FS / 2 * 0.95, SHIPPING, wind)),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=FONTSIZE, color=colour)

    ax.plot([], [], color="#52514e", lw=2.0, label="Generated noise (PSD)")
    ax.plot([], [], color="#52514e", lw=1.0, ls=(0, (4, 3)), alpha=0.75, label="Model")
    leg = ax.legend(loc="lower left", frameon=False, fontsize=FONTSIZE)
    for txt in leg.get_texts():
        txt.set_color("#52514e")

    ax.set_xlabel("Frequency [Hz]", fontsize=FONTSIZE, color="#52514e")
    ax.set_ylabel("PSD [dB re 1 µPa²/Hz]", fontsize=FONTSIZE, color="#52514e")
    ax.set_xlim(90, FS / 2 * 5.2)
    ax.set_ylim(0, 80)
    ax.grid(True, which="major", lw=0.6, zorder=1,ls = "--")
    ax.grid(True, which="minor", lw=0.35, alpha=0.6, zorder=1, ls = "--")
    ax.tick_params(colors="#52514e", labelsize=FONTSIZE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig("ambient_noise_psd.png", dpi=900, bbox_inches="tight")

    print(f"Band level over {F_LOW:.0f}-{F_HIGH:.0f} Hz (shipping {SHIPPING}):")
    for label, lvl in rows:
        print(f"  {label:<22s} {lvl:6.1f} dB re 1 uPa")


if __name__ == "__main__":
    main()