# BASE_DIR = "../dataset"
BASE_DIR = "/home/yusukemh/github/yusukemh/StatisticalDownscaling/dataset"

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

LABELS = [
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

ATTRIBUTES = [
    "air",
    "air",
    "hgt",
    "hgt",
    "omega",
    "pottmp",
    "pottmp",
    "pr_wtr",
    "shum",
    "shum",
    "shum",
    "shum",
    "shum",
    "shum",
    "skt",
    "slp"
]

BEST_MODEL_COLUMNS = [
    'air2m_0', 'air1000_500_0', 'hgt500_0', 'hgt1000_0', 'omega500_0', 'pottemp1000-500_0', 'pottemp1000-850_0', 
    'pr_wtr_0', 'shum-uwnd-700_0', 'shum-uwnd-925_0', 'shum-vwnd-700_0', 'shum-vwnd-950_0', 'shum700_0', 'shum925_0',
    'skt_0', 'slp_0', 'air2m_1', 'air1000_500_1', 'hgt500_1', 'hgt1000_1', 'omega500_1', 'pottemp1000-500_1',
    'pottemp1000-850_1', 'pr_wtr_1', 'shum-uwnd-700_1', 'shum-uwnd-925_1', 'shum-vwnd-700_1', 'shum-vwnd-950_1',
    'shum700_1', 'shum925_1', 'skt_1', 'slp_1', 'air2m_2', 'air1000_500_2', 'hgt500_2', 'hgt1000_2', 'omega500_2',
    'pottemp1000-500_2', 'pottemp1000-850_2', 'pr_wtr_2', 'shum-uwnd-700_2', 'shum-uwnd-925_2', 'shum-vwnd-700_2',
    'shum-vwnd-950_2', 'shum700_2', 'shum925_2', 'skt_2', 'slp_2', 'air2m_3', 'air1000_500_3', 'hgt500_3', 'hgt1000_3',
    'omega500_3', 'pottemp1000-500_3', 'pottemp1000-850_3', 'pr_wtr_3', 'shum-uwnd-700_3', 'shum-uwnd-925_3',
    'shum-vwnd-700_3', 'shum-vwnd-950_3', 'shum700_3', 'shum925_3', 'skt_3', 'slp_3', 'air2m_4', 'air1000_500_4',
    'hgt500_4', 'hgt1000_4', 'omega500_4', 'pottemp1000-500_4', 'pottemp1000-850_4', 'pr_wtr_4', 'shum-uwnd-700_4',
    'shum-uwnd-925_4', 'shum-vwnd-700_4', 'shum-vwnd-950_4', 'shum700_4', 'shum925_4', 'skt_4', 'slp_4', 'air2m_5',
    'air1000_500_5', 'hgt500_5', 'hgt1000_5', 'omega500_5', 'pottemp1000-500_5', 'pottemp1000-850_5', 'pr_wtr_5',
    'shum-uwnd-700_5', 'shum-uwnd-925_5', 'shum-vwnd-700_5', 'shum-vwnd-950_5', 'shum700_5', 'shum925_5', 'skt_5',
    'slp_5', 'data_in', 'lat', 'lon', 'elevation', 'season_wet', 'season_dry',
]

ISLAND_RANGES = [
    {
        "name": "kauai",
        "lat": (22.066281-0.45, 22.066281+0.45),
        "lon": (-159.526021-0.45, -159.526021+0.45)
    },
    {
        "name": "oahu",
        "lat": (21.485495-0.45, 21.485495+0.45),
        "lon": (-157.966174-0.45, -157.966174+0.45)
    },
    {
        "name": "molokai",
        "lat": (21.134806-0.15, 21.134806+0.15),
        "lon":  (-157.015431-0.35, -157.015431+0.35),
    },
    {
        "name": "lanai",
        "lat": (20.829217-0.1, 20.829217+0.1),
        "lon": (-156.926489-0.15, -156.926489+0.15),
    },
    {
        "name": "maui",
        "lat": (20.820998-0.24, 20.820998+0.24),
        "lon": (-156.312097-0.45, -156.312097+0.45)
    },
    {
        "name": "kahoolawe",
        "lat": (20.548690-0.1, 20.548690+0.1),
        "lon": (-156.608597-0.1, -156.608597+0.1)
    },
    {
        "name": "hawaii",
        "lat": (19.602708-0.7, 19.602708+0.7),
        "lon": (-155.474286-0.7, -155.474286+0.7)
    }
]

# common variables that needs to be loaded
C_COMMON = ['skn', 'data_in', 'season_wet', 'elevation', 'year', 'month', 'lat', 'lon']

# single closest grid cell
C_SINGLE = []
for item in LABELS:
    C_SINGLE.append(item)
C_SINGLE.extend(['elevation', 'season_wet', 'lat', 'lon'])

# interpolation: lower resolution
C_INT50 = []
for item in LABELS:
    C_INT50.append(f'i50_{item}')
C_INT50.extend(['elevation', 'season_wet', 'lat', 'lon'])

# interpolation: higher resolution
C_INT100 = []
for item in LABELS:
    C_INT100.append(f'i100_{item}')
C_INT100.extend(['elevation', 'season_wet', 'lat', 'lon'])

# grid data
C_GRID = []
for i in range(6):
    for item in LABELS:
        C_GRID.append(f"{item}_{i}")
C_GRID.extend(['elevation', 'season_wet', 'lat', 'lon'])
