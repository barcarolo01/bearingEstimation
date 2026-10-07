import os
from dotenv import load_dotenv
import xarray as xr

def get_depth_gebco(lat_query,lon_query):
    load_dotenv()
    HYDROMATE_PY_PATH = os.getenv('HYDROMATE_PY_PATH')
    PATH = os.path.join(HYDROMATE_PY_PATH,"gebco_2024_sub_ice_topo/GEBCO_2024_sub_ice_topo.nc")


    ds = xr.open_dataset(PATH, engine="netcdf4")
    elev = ds.elevation.sel(lon=lon_query,lat=lat_query,method="nearest")

    lon_actual = float(elev.lon)
    lat_actual = float(elev.lat)
    elev_val   = int(elev.values)
    return elev_val * (-1) # Positive value = underwater


if __name__ == '__main__':
    lat_query = 20.832813
    lon_query =  88.698390
    
    print(get_depth_gebco(lat_query,lon_query))