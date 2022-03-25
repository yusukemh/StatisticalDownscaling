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
FILE_NAME = './report.txt'

def report(message, report_time=False, start=None):
    with open(FILE_NAME, 'a') as f:
        f.write(message + '\n')
        if report_time:
            curr = time.time()
            f.write("Elapsed time: {:.2f}\n".format(curr - start))

def main():
    start = time.time()
    report('this is updated code')
    report("starting CV")
    

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

    df_combined = pd.concat([df_train, df_valid, df_test])
        
    report('dataset loaded')
    report('running linear regression')
        
    # Linear regression
    n_cv = 5
    dfs = []
    for i, (name, group) in enumerate(df_combined.groupby(by="skn")):
        X = np.array(group[columns].drop("data_in", axis=1))
        Y = np.array(group["data_in"])
        if X.shape[0] < n_cv:
            group['prediction_multi_linear'] = [np.nan] * X.shape[0]
            dfs.append(group)
            continue

        yhat = cross_val_predict(LinearRegression(), X, Y, cv=n_cv, n_jobs=-1)
        group["prediction_multi_linear"] = yhat
        dfs.append(group)
    
    df_multi_linear = pd.concat(dfs)
    df_multi_linear[['skn', 'year', 'month', 'data_in', 'name', 'season_dry', 'season_wet', 'prediction_multi_linear']].to_csv(f"{BASE_DIR}/cv/multi_linear.csv", index=False)
    
    report('linear regression complete', report_time=True, start=start)

    # XGBoost
    n_cv = 5
    dfs = []
    num_groups = df_combined['skn'].unique().shape[0]
    report(f'xgb initiating. There are {num_groups} stations')

    for i, (name, group) in enumerate(df_combined.groupby(by="skn")):
        X = np.array(group[columns].drop("data_in", axis=1))
        Y = np.array(group["data_in"])
        if X.shape[0] < n_cv: 
            group['prediction_multi_xgb'] = [np.nan] * X.shape[0]
            dfs.append(group)
            continue

        xgboost = XGBRegressor(
            n_estimators=170,
            learning_rate=0.1,
            max_depth=9,
            verbosity=0
        )

        yhat = cross_val_predict(xgboost, X, Y, cv=n_cv, n_jobs=-1)
        group['prediction_multi_xgb'] = yhat
        dfs.append(group)
        
        print(f"{i}/{num_groups}")
        
    df_multi_xgb = pd.concat(dfs)
    df_multi_xgb[['skn', 'year', 'month', 'data_in', 'name', 'season_dry', 'season_wet', 'prediction_multi_linear']].to_csv(f"{BASE_DIR}/cv/multi_xgb.csv", index=False)
    
    report('xgb complete', report_time=True, start=start)
    
    
    
if __name__ == "__main__":
    main()