import math

def sposta(lat,lon,distanza,dir):

    # conversioni
    delta_lat = distanza / 111320
    delta_lon = distanza / (111320 * math.cos(math.radians(lat)))


    if dir == "N":
        new_lat = lat + delta_lat # NORD
        new_lon = lon
    if dir == "S":
        new_lat = lat - delta_lat # SUD
        new_lon = lon
    if dir == "E":
        new_lat = lat 
        new_lon = lon + delta_lon
    if dir == "O":
        new_lat = lat 
        new_lon = lon - delta_lon


    return new_lat, new_lon