PATH = "C:/Users/Nicola/Desktop/TESI/HYDROMATE/hm_code/gebco_2024_sub_ice_topo/GEBCO_2024_sub_ice_topo.nc"

import xarray as xr

lon_query = -73.546610
lat_query = 39.839150

ds = xr.open_dataset(PATH, engine="netcdf4")

elev = ds.elevation.sel(
    lon=lon_query,
    lat=lat_query,
    method="nearest"
)

lon_actual = float(elev.lon)
lat_actual = float(elev.lat)
elev_val   = int(elev.values)

print(f"Punto richiesto : lon={lon_query}, lat={lat_query}")
print(f"Punto più vicino: lon={lon_actual:.4f}, lat={lat_actual:.4f}")
print(f"Elevazione      : {elev_val} m  ({'terra' if elev_val > 0 else 'mare' if elev_val < 0 else 'livello del mare'})")