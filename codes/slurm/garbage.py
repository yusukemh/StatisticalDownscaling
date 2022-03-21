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
    file_name = './garbage.txt'

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
    
    line = f"labels to use: {columns}\n"
    with open(file_name, 'a') as f:
        f.write(line)

    Xtrain = np.array(df_train[columns].drop(labels=["data_in"], axis=1))
    Ytrain = np.array(df_train["data_in"])

    Xvalid = np.array(df_valid[columns].drop(labels=["data_in"], axis=1))
    Yvalid = np.array(df_valid["data_in"])

    Xtest = np.array(df_test[columns].drop(labels=["data_in"], axis=1))
    Ytest = np.array(df_test["data_in"])
    
    gradient_boost = GradientBoostingRegressor(
        n_estimators=300, 
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=2,
        verbose=False
    )

    gradient_boost.fit(Xtrain, Ytrain)
    line = "MSE on gradient boost (test) : {:.3f}".format(mean_squared_error(Ytest, gradient_boost.predict(Xtest)))
    line += "MSE on gradient boost (train): {:.3f}".format(mean_squared_error(Ytrain, gradient_boost.predict(Xtrain)))
    


if __name__ == "__main__":
    main()