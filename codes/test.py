import numpy as np
import netCDF4 as nc
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from datetime import datetime
import xarray as xr
import sys
from time import time
# from mpl_toolkits.basemap import Basemap

base_dir = "../dataset"
# nc.Dataset(f"{base_dir}/")
df_locations = pd.read_csv(f"{base_dir}/SKNlocations.csv")
df_data = pd.read_excel(f"{base_dir}/FilledDataset2012.xlsx", sheet_name="Data_in")

X = []
for index, row in df_data.iterrows():
    if row.Year < 1948:
        # No need to keep data older than 1948 becase no data exists in netCDF files
        continue
    for i, cell in enumerate(row[2:]):
        X.append([row.SKN, row.Year, i + 1, cell])

df_data_by_cell = pd.DataFrame(X, columns = ["skn", "year", "month", "data_in"]).dropna()
df_data_by_cell = df_data_by_cell.replace(r'^\s*$', np.nan, regex=True).dropna()

df_data_w_coord = df_data_by_cell.merge(right=df_locations, left_on="skn", right_on="SKN")



# ds = xr.open_dataset(f"{base_dir}/air.2m.mon.mean.regridded.nc")
lat_hawaii = [15, 17.5, 20, 22.5, 25]
lon_hawaii = np.array([-162.5, -160, -157.5, -155, -152.5]) + 360

ds_air2m = xr.open_dataset(f"{base_dir}/air.2m.mon.mean.regridded.nc").loc[dict(lat=lat_hawaii, lon=lon_hawaii)]
ds_air1000_500 = xr.open_dataset(f"{base_dir}/air.1000-500.mon.mean.nc").loc[dict(lat=lat_hawaii, lon=lon_hawaii)]
ds_hgt500 = xr.open_dataset(f"{base_dir}/hgt500.mon.mean.nc").loc[dict(lat=lat_hawaii, lon=lon_hawaii)]
ds_hgt1000 = xr.open_dataset(f"{base_dir}/hgt1000.mon.mean.nc").loc[dict(lat=lat_hawaii, lon=lon_hawaii)]
ds_omega500 = xr.open_dataset(f"{base_dir}/omega500.mon.mean.nc").loc[dict(lat=lat_hawaii, lon=lon_hawaii)]
ds_pottemp_1000_500 = xr.open_dataset(f"{base_dir}/pottmp.1000-500.mon.mean.nc").loc[dict(lat=lat_hawaii, lon=lon_hawaii)]
ds_pottemp_1000_850 = xr.open_dataset(f"{base_dir}/pottmp.1000-850.mon.mean.nc").loc[dict(lat=lat_hawaii, lon=lon_hawaii)]
ds_pwtr = xr.open_dataset(f"{base_dir}/pwtr.mon.mean.nc").loc[dict(lat=lat_hawaii, lon=lon_hawaii)]
ds_u700 = xr.open_dataset(f"{base_dir}/shum_x_uwnd.700.mon.mean.nc").loc[dict(lat=lat_hawaii, lon=lon_hawaii)]
ds_u925 = xr.open_dataset(f"{base_dir}/shum_x_uwnd.925.mon.mean.nc").loc[dict(lat=lat_hawaii, lon=lon_hawaii)]
ds_v700 = xr.open_dataset(f"{base_dir}/shum_x_vwnd.700.mon.mean.nc").loc[dict(lat=lat_hawaii, lon=lon_hawaii)]
ds_v950 = xr.open_dataset(f"{base_dir}/shum_x_vwnd.925.mon.mean.nc").loc[dict(lat=lat_hawaii, lon=lon_hawaii)]
ds_shum700 = xr.open_dataset(f"{base_dir}/shum700.mon.mean.nc").loc[dict(lat=lat_hawaii, lon=lon_hawaii)]
ds_shum925 = xr.open_dataset(f"{base_dir}/shum925.mon.mean.nc").loc[dict(lat=lat_hawaii, lon=lon_hawaii)]
ds_skt = xr.open_dataset(f"{base_dir}/skt.mon.mean.regridded.nc").loc[dict(lat=lat_hawaii, lon=lon_hawaii)]
ds_slp = xr.open_dataset(f"{base_dir}/slp.mon.mean.nc").loc[dict(lat=lat_hawaii, lon=lon_hawaii)]

datasets = [ # list of tuples. (dataset object, label, attribute string in ds)
    (ds_air2m, "air2m", "air"),
    (ds_air1000_500, "air1000_500", "air"),
    (ds_hgt500, "hgt500", "hgt"),
    (ds_hgt1000, "hgt1000", "hgt"),
    (ds_omega500, "omega500", "omega"),
    (ds_pottemp_1000_500, "pottemp1000-500", "pottmp"),
    (ds_pottemp_1000_850, "pottemp1000-850", "pottmp"),
    (ds_pwtr, "pr_wtr", "pr_wtr"),
    (ds_u700, "shum-uwnd-700", "shum"),
    (ds_u925, "shum-uwnd-925", "shum"),
    (ds_v700, "shum-vwnd-700", "shum"),
    (ds_v950, "shum-vwnd-950", "shum"),
    (ds_shum700, "shum700", "shum"),
    (ds_shum925, "shum925", "shum"),
    (ds_skt, "skt", "skt"),
    (ds_slp, "slp", "slp")
]
# combine all the cdf data

start = time()
df_data_xarray = pd.DataFrame()

for dataset in datasets:
    array = dataset[0]
    label = dataset[1]
    df_data_xarray[label] = df_data_w_coord[:10].apply(lambda x: array.loc[dict(time=f"{x['year']}-{x['month']}-01")].to_array(), axis=1)
end = time()
print(end-start)

df_data_xarray.to_csv(f"{base_dir}/temp.csv")