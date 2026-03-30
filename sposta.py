import math

lat = -28.890373
lon = 113.897648

distanza = 1000  # metri

# conversioni
delta_lat = distanza / 111320
delta_lon = distanza / (111320 * math.cos(math.radians(lat)))

# spostamento verso nord (solo latitudine)
new_lat = lat + delta_lat
new_lon = lon #+ delta_lon

print(new_lat, new_lon)