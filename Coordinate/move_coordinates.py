import math

# Parametri ellissoide WGS-84 (standard GPS)
WGS84_A = 6_378_137.0          # semiasse maggiore (m)
WGS84_F = 1 / 298.257223563    # schiacciamento
WGS84_B = WGS84_A * (1 - WGS84_F)  # semiasse minore (m)


def sposta_coordinate_vincenty(lat1: float, lon1: float, distanza_m: float, bearing_deg: float):
    """
    Calcola il punto di arrivo dato un punto di partenza, una distanza e un azimut
    usando la formula diretta di Vincenty sull'ellissoide WGS-84.

    Args:
        lat1        : Latitudine di partenza in gradi decimali  (es. 40.6405)
        lon1        : Longitudine di partenza in gradi decimali (es. 14.9897)
        distanza_m  : Distanza di spostamento in METRI          (es. 0.30 per 30 cm)
        bearing_deg : Azimut/direzione in gradi                 (0=N, 90=E, 180=S, 270=O)

    Returns:
        (latitudine2, longitudine2) in gradi decimali
    """
    a = WGS84_A
    f = WGS84_F
    b = WGS84_B

    phi1   = math.radians(lat1)
    lam1   = math.radians(lon1)
    alpha1 = math.radians(bearing_deg)
    s      = distanza_m

    sin_alpha1 = math.sin(alpha1)
    cos_alpha1 = math.cos(alpha1)

    # Latitudine ridotta (sulla sfera ausiliaria)
    tan_U1 = (1 - f) * math.tan(phi1)
    cos_U1 = 1 / math.sqrt(1 + tan_U1 ** 2)
    sin_U1 = tan_U1 * cos_U1

    # Prima equazione angolare
    sigma1 = math.atan2(tan_U1, cos_alpha1)

    # Azimut all'equatore
    sin_alpha = cos_U1 * sin_alpha1
    cos2_alpha = 1 - sin_alpha ** 2

    u2 = cos2_alpha * (a ** 2 - b ** 2) / (b ** 2)

    A_coeff = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B_coeff = u2 / 1024  * (256  + u2 * (-128 + u2 * (74  - 47  * u2)))

    sigma = s / (b * A_coeff)

    for _ in range(1000):  # iterazione fino a convergenza
        cos2_sigma_m = math.cos(2 * sigma1 + sigma)
        sin_sigma    = math.sin(sigma)
        cos_sigma    = math.cos(sigma)

        delta_sigma = (
            B_coeff * sin_sigma * (
                cos2_sigma_m
                + B_coeff / 4 * (
                    cos_sigma * (-1 + 2 * cos2_sigma_m ** 2)
                    - B_coeff / 6
                    * cos2_sigma_m
                    * (-3 + 4 * sin_sigma ** 2)
                    * (-3 + 4 * cos2_sigma_m ** 2)
                )
            )
        )
        sigma_new = s / (b * A_coeff) + delta_sigma

        if abs(sigma_new - sigma) < 1e-12:   # ~0.06 mm di precisione
            break
        sigma = sigma_new

    cos2_sigma_m = math.cos(2 * sigma1 + sigma)
    sin_sigma    = math.sin(sigma)
    cos_sigma    = math.cos(sigma)

    # Latitudine di arrivo
    phi2 = math.atan2(
        sin_U1 * cos_sigma + cos_U1 * sin_sigma * cos_alpha1,
        (1 - f) * math.sqrt(sin_alpha ** 2 + (sin_U1 * sin_sigma - cos_U1 * cos_sigma * cos_alpha1) ** 2)
    )

    # Differenza di longitudine sulla sfera ausiliaria
    lam = math.atan2(
        sin_sigma * sin_alpha1,
        cos_U1 * cos_sigma - sin_U1 * sin_sigma * cos_alpha1
    )

    # Correzione C
    C = f / 16 * cos2_alpha * (4 + f * (4 - 3 * cos2_alpha))

    # Differenza di longitudine sull'ellissoide
    L = lam - (1 - C) * f * sin_alpha * (
        sigma + C * sin_sigma * (
            cos2_sigma_m + C * cos_sigma * (-1 + 2 * cos2_sigma_m ** 2)
        )
    )

    lon2 = lam1 + L

    lat2 = math.degrees(phi2)
    lon2 = math.degrees(lon2)

    return lat2, lon2


def direzione_a_bearing(direzione: str) -> float:
    """
    Converte una direzione cardinale (o un angolo numerico) in gradi azimutali.

    Direzioni accettate:
        N, Nord          ->   0°
        NE, Nord-Est     ->  45°
        E, Est           ->  90°
        SE, Sud-Est      -> 135°
        S, Sud           -> 180°
        SO, Sud-Ovest    -> 225°
        O, Ovest         -> 270°
        NO, Nord-Ovest   -> 315°
        oppure un numero float (es. 37.5) già in gradi azimutali
    """
    mappa = {
        "N": 0.0, "NORD": 0.0,
        "NE": 45.0, "NORDEST": 45.0, "NORD-EST": 45.0,
        "E": 90.0, "EST": 90.0,
        "SE": 135.0, "SUDEST": 135.0, "SUD-EST": 135.0,
        "S": 180.0, "SUD": 180.0,
        "SO": 225.0, "SUDOVEST": 225.0, "SUD-OVEST": 225.0,
        "O": 270.0, "OVEST": 270.0,
        "NO": 315.0, "NORDOVEST": 315.0, "NORD-OVEST": 315.0,
    }
    chiave = direzione.strip().upper()
    if chiave in mappa:
        return mappa[chiave]
    try:
        return float(chiave)
    except ValueError:
        raise ValueError(
            f"Direzione '{direzione}' non riconosciuta.\n"
            "Usa una direzione cardinale (N, NE, E, SE, S, SO, O, NO) "
            "oppure un angolo in gradi (es. 45.0)."
        )


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    #  MODIFICA QUESTE VARIABILI                                          #
    # ------------------------------------------------------------------ #
    latitudine1  = -35.9570   # gradi decimali
    longitudine1 = 153.3208  # gradi decimali
    distanza     = 200     # METRI  (es. 0.30 = 30 cm)
    direzione    = "E"         # N, NE, E, SE, S, SO, O, NO  oppure gradi (es. "37.5")
    # ------------------------------------------------------------------ #

    bearing = direzione_a_bearing(direzione)

    latitudine2, longitudine2 = sposta_coordinate_vincenty(
        latitudine1, longitudine1, distanza, bearing
    )

    print("=" * 52)
    print("  Spostamento coordinate  —  Formula di Vincenty")
    print("  Ellissoide WGS-84  |  precisione < 0.1 mm")
    print("=" * 52)
    print(f"  Punto di partenza : {latitudine1:.10f}, {longitudine1:.10f}")
    print(f"  Direzione         : {direzione}  ({bearing}°)")
    print(f"  Distanza          : {distanza} m")
    print(f"  Punto di arrivo   : {latitudine2:.10f}, {longitudine2:.10f}")
    print("=" * 52)