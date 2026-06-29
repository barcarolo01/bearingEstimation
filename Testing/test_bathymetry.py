import xarray as xr

PATH = "C:/Users/Nicola/Desktop/TESI/HYDROMATE/hm_code/gebco_2024_sub_ice_topo/GEBCO_2024_sub_ice_topo.nc"

lon_query = 55.141862749665
lat_query = 25.903322371762606

ds = xr.open_dataset(PATH, engine="netcdf4")

elev = ds.elevation.sel(
    lon=lon_query,
    lat=lat_query,
    method="nearest"
)

lon_actual = float(elev.lon)
lat_actual = float(elev.lat)
elev_val   = int(elev.values)

print(f"Queried point : lon={lon_query}, lat={lat_query}")
print(f"Closest point: lon={lon_actual:.4f}, lat={lat_actual:.4f}")
print(f"Elevation: {elev_val} m  ")