# CHECK FILE NAME, MODEL, PARAMETER SPACE
# basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans

#others
from xgboost import XGBRegressor
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import time
import xarray as xr
import sherpa

BASE_DIR='/home/yusukemh/github/yusukemh/StatisticalDownscaling/dataset'

def main():
    file_name = './GBR_interp100.txt'

    reanalysis_data = [
        'air2m', 'air1000_500', 'hgt500', 'hgt1000', 'omega500',
        'pottemp1000-500', 'pottemp1000-850', 'pr_wtr', 'shum-uwnd-700',
        'shum-uwnd-925', 'shum-vwnd-700', 'shum-vwnd-950', 'shum700', 'shum925', 
        'skt', 'slp'
    ]

    columns = []
    for i in range(6):
        for item in reanalysis_data:
            columns.append(f"{item}_{i}")
    columns.extend(['data_in', 'lat', 'lon', 'elevation', 'season_wet', 'season_dry'])
    print(f"columns to use: {columns}")
    
    # load datasets
    df_train = pd.read_csv(f"{BASE_DIR}/train.csv")
    df_valid = pd.read_csv(f"{BASE_DIR}/valid.csv")
    df_test = pd.read_csv(f"{BASE_DIR}/test.csv")
    
    # Nov-Apr = "wet", May-Oct = "dry"
    wet = [11, 12, 1, 2, 3, 4]
    dry = [5, 6, 7, 8, 9, 10]
    df_train['season_dry'] = df_train.apply(lambda row: 1 if row.month in dry else 0, axis=1)
    df_train['season_wet'] = df_train.apply(lambda row: 1 if row.month in wet else 0, axis=1)

    df_valid['season_dry'] = df_valid.apply(lambda row: 1 if row.month in dry else 0, axis=1)
    df_valid['season_wet'] = df_valid.apply(lambda row: 1 if row.month in wet else 0, axis=1)

    df_test['season_dry'] = df_test.apply(lambda row: 1 if row.month in dry else 0, axis=1)
    df_test['season_wet'] = df_test.apply(lambda row: 1 if row.month in wet else 0, axis=1)
    

    df_combined = pd.concat([df_train, df_valid, df_test])
    
    
    
    # XGBoost
    # make sure to groupby by skn: some stations have the same name e.g., WAIMEA
    ytrue = []
    ypred = []

    n_data = []
    rmse_per_station = []

    n_cv = 5

    num_groups = df_combined['skn'].unique().shape[0]

    for i, (name, group) in enumerate(df_combined.groupby(by="skn")):
        X = np.array(group[columns].drop("data_in", axis=1))
        Y = np.array(group["data_in"])
        if X.shape[0] < n_cv: continue

        xgboost = XGBRegressor(
            n_estimators=170,
            learning_rate=0.1,
            max_depth=9,
            verbosity=0
        )


        yhat = cross_val_predict(xgboost, X, Y, cv=n_cv, n_jobs=-1)
        rmse = mean_squared_error(Y, yhat, squared=False)

        ytrue.extend(Y)
        ypred.extend(yhat)

        n_data.append(X.shape[0])
        rmse_per_station.append(rmse)

        print(f"{i}/{num_groups}")
    np.savetxt("ytrue.csv", np.array(ytrue), delimiter=",")
    np.savetxt("yperd.csv", np.array(ypred), delimiter=",")
    np.savetxt("n_data.csv", np.array(n_data), delimiter=",")
    np.savetxt("rmse_per_station.csv", np.array(rmse_per_station), delimiter=",")
    
    
if __name__ == "__main__":
    main()