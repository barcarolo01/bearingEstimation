"""
find_intersection_points.py

Calcola i punti di intersezione tra semirette geodetiche sulla superficie terrestre.
Gli angoli sono espressi in gradi, misurati in senso ANTIORARIO a partire dalla direzione EST
(convenzione matematica standard).

Uso:
    H1 = (-35, 153)
    H2 = (-33, 151)
    bearing_H1 = [45, 90, 135]
    bearing_H2 = [225, 270, 315]
    result = find_points(H1, H2, bearing_H1, bearing_H2)
    # result è un array Nx2 con colonne [lat, lon]
"""

import numpy as np


# ---------------------------------------------------------------------------
# Utilità angolari
# ---------------------------------------------------------------------------

def math_to_bearing(math_angle_deg: float) -> float:
    """
    Converte un angolo matematico (antiorario da EST, in gradi)
    nel bearing geografico (orario da NORD, in gradi) usato internamente
    per i calcoli geodetici.

    Relazione:  bearing = 90 - math_angle   (mod 360)
    """
    return (90.0 - math_angle_deg) % 360.0


# ---------------------------------------------------------------------------
# Intersezione di due grandi cerchi  (coordinate sferiche)
# ---------------------------------------------------------------------------

def _great_circle_intersection(
    lat1: float, lon1: float, brg1: float,
    lat2: float, lon2: float, brg2: float,
) -> tuple[float, float] | tuple[float, float]:
    """
    Restituisce il punto di intersezione tra due semirette geodetiche.

    Parametri
    ---------
    lat1, lon1 : coordinate del primo punto (gradi decimali)
    brg1       : bearing geografico (°N, orario) della prima semiretta
    lat2, lon2 : coordinate del secondo punto (gradi decimali)
    brg2       : bearing geografico (°N, orario) della seconda semiretta

    Ritorna
    -------
    (lat, lon) del punto di intersezione, oppure (nan, nan) se non esiste.

    Algoritmo
    ---------
    Implementazione vettoriale basata sul metodo di Ed Williams
    (Aviation Formulary, intersection of two radials):
    https://edwilliams.org/avform147.htm#Intersection
    """
    # Converti tutto in radianti
    φ1, λ1 = np.radians(lat1), np.radians(lon1)
    φ2, λ2 = np.radians(lat2), np.radians(lon2)
    θ13    = np.radians(brg1)
    θ23    = np.radians(brg2)

    Δφ = φ2 - φ1
    Δλ = λ2 - λ1

    # Distanza angolare tra i due punti di origine
    δ12 = 2.0 * np.arcsin(
        np.sqrt(
            np.sin(Δφ / 2.0) ** 2
            + np.cos(φ1) * np.cos(φ2) * np.sin(Δλ / 2.0) ** 2
        )
    )

    if np.abs(δ12) < 1e-12:           # punti coincidenti
        return np.nan, np.nan

    # Bearing iniziale e finale del segmento 1→2
    cos_θa = (np.sin(φ2) - np.sin(φ1) * np.cos(δ12)) / (np.sin(δ12) * np.cos(φ1))
    cos_θb = (np.sin(φ1) - np.sin(φ2) * np.cos(δ12)) / (np.sin(δ12) * np.cos(φ2))

    cos_θa = np.clip(cos_θa, -1.0, 1.0)
    cos_θb = np.clip(cos_θb, -1.0, 1.0)

    θa = np.arccos(cos_θa)
    θb = np.arccos(cos_θb)

    if np.sin(Δλ) > 0:
        θ12, θ21 = θa, 2.0 * np.pi - θb
    else:
        θ12, θ21 = 2.0 * np.pi - θa, θb

    α1 = θ13 - θ12          # angolo al vertice 1
    α2 = θ21 - θ23          # angolo al vertice 2

    # Angoli quasi paralleli → nessuna intersezione utile
    if np.abs(np.sin(α1)) < 1e-12 and np.abs(np.sin(α2)) < 1e-12:
        return np.nan, np.nan

    α3 = np.arccos(
        np.clip(
            -np.cos(α1) * np.cos(α2)
            + np.sin(α1) * np.sin(α2) * np.cos(δ12),
            -1.0, 1.0,
        )
    )

    δ13 = np.arctan2(
        np.sin(δ12) * np.sin(α1) * np.sin(α2),
        np.cos(α2) + np.cos(α1) * np.cos(α3),
    )

    φ3 = np.arcsin(
        np.clip(
            np.sin(φ1) * np.cos(δ13)
            + np.cos(φ1) * np.sin(δ13) * np.cos(θ13),
            -1.0, 1.0,
        )
    )

    Δλ13 = np.arctan2(
        np.sin(θ13) * np.sin(δ13) * np.cos(φ1),
        np.cos(δ13) - np.sin(φ1) * np.sin(φ3),
    )

    λ3 = λ1 + Δλ13

    lat_i = np.degrees(φ3)
    lon_i = (np.degrees(λ3) + 540.0) % 360.0 - 180.0

    return lat_i, lon_i


def _point_is_on_semiretta(
    lat0: float, lon0: float, bearing_geo: float,
    lat_i: float, lon_i: float,
    tol_deg: float = 0.5,
) -> bool:
    """
    Verifica che il punto (lat_i, lon_i) si trovi sulla semiretta
    (non sul prolungamento opposto) definita da (lat0, lon0) con
    bearing geografico bearing_geo.

    Il test è: il bearing dal punto di origine verso il punto di intersezione
    deve essere entro tol_deg dal bearing atteso.
    """
    φ0, λ0 = np.radians(lat0), np.radians(lon0)
    φi, λi = np.radians(lat_i), np.radians(lon_i)
    Δλ = λi - λ0

    brg_to_i = np.degrees(
        np.arctan2(
            np.sin(Δλ) * np.cos(φi),
            np.cos(φ0) * np.sin(φi) - np.sin(φ0) * np.cos(φi) * np.cos(Δλ),
        )
    ) % 360.0

    diff = abs((brg_to_i - bearing_geo + 180.0) % 360.0 - 180.0)
    return diff <= tol_deg


# ---------------------------------------------------------------------------
# Funzione pubblica
# ---------------------------------------------------------------------------

def find_points(
    H1: tuple[float, float],
    H2: tuple[float, float],
    bearing_H1: list[float] | np.ndarray,
    bearing_H2: list[float] | np.ndarray,
) -> np.ndarray:
    """
    Calcola i punti di intersezione tra coppie di semirette geodetiche.

    Parametri
    ---------
    H1 : (lat, lon) del primo punto origine, in gradi decimali.
    H2 : (lat, lon) del secondo punto origine, in gradi decimali.
    bearing_H1 : array di N angoli (gradi) per H1,
                 misurati in senso ANTIORARIO a partire da EST.
    bearing_H2 : array di N angoli (gradi) per H2,
                 misurati in senso ANTIORARIO a partire da EST.

    Ritorna
    -------
    np.ndarray di forma (N, 2) con colonne [latitudine, longitudine].
    Le righe senza intersezione valida contengono [NaN, NaN].
    """
    bearing_H1 = np.asarray(bearing_H1, dtype=float)
    bearing_H2 = np.asarray(bearing_H2, dtype=float)

    if bearing_H1.shape != bearing_H2.shape:
        raise ValueError(
            f"bearing_H1 e bearing_H2 devono avere la stessa dimensione, "
            f"ma sono {bearing_H1.shape} e {bearing_H2.shape}."
        )

    n = bearing_H1.size
    result = np.full((n, 2), np.nan)

    lat1, lon1 = float(H1[0]), float(H1[1])
    lat2, lon2 = float(H2[0]), float(H2[1])

    for k in range(n):
        # Converti dalla convenzione matematica (CCW da EST)
        # al bearing geografico (CW da NORD)
        brg1 = math_to_bearing(bearing_H1[k])
        brg2 = math_to_bearing(bearing_H2[k])

        lat_i, lon_i = _great_circle_intersection(
            lat1, lon1, brg1,
            lat2, lon2, brg2,
        )

        if np.isnan(lat_i):
            continue  # nessuna intersezione geometrica

        # Verifica che il punto cada su entrambe le SEMI-rette
        # (non sul prolungamento opposto del raggio)
        on1 = _point_is_on_semiretta(lat1, lon1, brg1, lat_i, lon_i)
        on2 = _point_is_on_semiretta(lat2, lon2, brg2, lat_i, lon_i)

        if on1 and on2:
            result[k, 0] = lat_i
            result[k, 1] = lon_i
        # altrimenti rimane NaN

    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Esempio: due stazioni radar in Australia orientale
    H1 = (-35.0, 149.0)   # Canberra area
    H2 = (-33.9, 151.2)   # Sydney area

    # Angoli in gradi, senso antiorario da EST (convezione matematica)
    # 0°  = EST, 90° = NORD, 180° = OVEST, 270° = SUD
    bearing_H1 = [90.0,  80.0,  70.0]   # punta verso nord-est circa
    bearing_H2 = [110.0, 100.0, 120.0]  # punta verso nord-ovest circa

    intersections = find_points(H1, H2, bearing_H1, bearing_H2)

    print("Punti di intersezione [lat, lon]:")
    for i, (lat, lon) in enumerate(intersections):
        if np.isnan(lat):
            print(f"  [{i}]  nessuna intersezione valida")
        else:
            print(f"  [{i}]  lat={lat:.5f}°  lon={lon:.5f}°")