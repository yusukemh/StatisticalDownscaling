FILE_NAMES = [
    "air.2m.mon.mean.regridded.nc",
    "air.1000-500.mon.mean.nc",
    "hgt500.mon.mean.nc",
    "hgt1000.mon.mean.nc",
    "omega500.mon.mean.nc",
    "pottmp.1000-500.mon.mean.nc",
    "pottmp.1000-850.mon.mean.nc",
    "pwtr.mon.mean.nc",
    "shum_x_uwnd.700.mon.mean.nc",
    "shum_x_uwnd.925.mon.mean.nc",
    "shum_x_vwnd.700.mon.mean.nc",
    "shum_x_vwnd.925.mon.mean.nc",
    "shum700.mon.mean.nc",
    "shum925.mon.mean.nc",
    "skt.mon.mean.regridded.nc",
    "slp.mon.mean.nc"
]

DF_LABELS = [
    "air2m",
    "air1000_500",
    "hgt500",
    "hgt1000",
    "omega500",
    "pottemp1000-500",
    "pottemp1000-850",
    "pr_wtr",
    "shum-uwnd-700",
    "shum-uwnd-925",
    "shum-vwnd-700",
    "shum-vwnd-950",
    "shum700",
    "shum925",
    "skt",
    "slp"
]

C_COMMON = [
    'skn', 'data_in', 'season_wet',
    'elevation', 'year', 'month', 'lat', 'lon'
]

C_SINGLE = []
for item in DF_LABELS:
    C_SINGLE.append(item)

C_GRID = []
for item in DF_LABELS:
    for i in range(5):
        C_GRID.append(f"{item}_{i}")
