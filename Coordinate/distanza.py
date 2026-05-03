import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calcola la distanza tra due punti sulla Terra (in metri)
    usando la formula di Haversine.
    """
    R = 6371000  # Raggio medio della Terra in metri

    # Conversione da gradi a radianti
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Differenze
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Formula di Haversine
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distanza = R * c
    return distanza


# Input utente
lat1 = float(input("Inserisci latitudine punto 1: "))
lon1 = float(input("Inserisci longitudine punto 1: "))
lat2 = float(input("Inserisci latitudine punto 2: "))
lon2 = float(input("Inserisci longitudine punto 2: "))

d = haversine(lat1, lon1, lat2, lon2)

print(f"La distanza tra i due punti è: {d:.2f} metri")