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
import time

BASE_DIR='/home/yusukemh/github/yusukemh/StatisticalDownscaling/dataset'

def main():
    file_name = './LOOCV.txt'

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
    line = f"labels to use: {columns}\n"
    with open(file_name, 'a') as f:
        f.write(line)

    station_candidates = []
    for name, group in df_train.groupby(by='name'):
        if group.shape[0] > 360:
            station_candidates.append(name)
    np.random.seed(42)
    stations = np.random.choice(station_candidates, size=5)
    
    start = time.time()
    results = []
    for station in stations:
        df_train_station = df_train[df_train['name'] == station]
        df_test_station  = df_test[df_test['name'] == station]
        if df_train_station.shape[0] == 0 or df_test_station.shape[0] == 0:
            continue
        print("=========================================")
        print(f"Running experiment on {station} station.")
        print(f"There are:")
        print(f"{df_train_station.shape[0]} training data and")
        print(f"{df_test_station.shape[0]} test data")
        print("=========================================")

        # xgboost trains on the entire dataset
        Xtrain = np.array(df_train[columns].drop(labels=["data_in"], axis=1))
        Ytrain = np.array(df_train["data_in"])
        # test on the station data
        Xtest = np.array(df_test_station[columns].drop(labels=["data_in"], axis=1))
        Ytest = np.array(df_test_station["data_in"])

        # hyperparameters obtained by fine tuning
        xgboost = XGBRegressor(
            n_estimators=170,
            learning_rate=0.1,
            max_depth=9,
            verbosity=0
        )

        xgboost.fit(Xtrain, Ytrain)
        print("MSE on xgboost (test) : {:.5f}".format(mean_squared_error(Ytest, xgboost.predict(Xtest))))
        print("MSE on xgboost (train): {:.5f}".format(mean_squared_error(Ytrain, xgboost.predict(Xtrain))))
        mse_xgb = mean_squared_error(Ytest, xgboost.predict(Xtest))

        # linear regression trains on the station dataset
        Xtrain = np.array(df_train_station[columns].drop(labels=["data_in"], axis=1))
        Ytrain = np.array(df_train_station["data_in"])
        # test on the station data
        Xtest = np.array(df_test_station[columns].drop(labels=["data_in"], axis=1))
        Ytest = np.array(df_test_station["data_in"])

        model = LinearRegression()
        model.fit(Xtrain, Ytrain)
        print("MSE on Linear Regression (test) : {:.5f}".format(mean_squared_error(Ytest, model.predict(Xtest))))
        print("MSE on Linear Regression (train): {:.5f}".format(mean_squared_error(Ytrain, model.predict(Xtrain))))
        mse_linear = mean_squared_error(Ytest, model.predict(Xtest))

        results.append(
            {"n_samples": df_train_station.shape[0],
             "MSE_xgb": mse_xgb,
             "MSE_linear": mse_linear
            }
        )

    sum_weighted_mse_xgb = 0
    sum_weighted_mse_linear = 0
    total_n_samples = 0
    for item in results:
        total_n_samples += item["n_samples"]
        sum_weighted_mse_xgb += item["MSE_xgb"] * item["n_samples"] 
        sum_weighted_mse_linear += item["MSE_linear"] * item["n_samples"] 

    mean_mse_xgb, mean_mse_linear = sum_weighted_mse_xgb/total_n_samples, sum_weighted_mse_linear/total_n_samples
    print("mean of MSE with XGBoost: {:.5f}".format(mean_mse_xgb))
    print("mean of MSE with Linear Regression: {:.5f}".format(mean_mse_linear))
    
    line = "mean of MSE with XGBoost: {:.5f}".format(mean_mse_xgb)
    line += "mean of MSE with Linear Regression: {:.5f}".format(mean_mse_linear)

    end = time.time()
    print(end - start)
    
    
    with open(file_name, 'a') as f:
        f.write(line)


if __name__ == "__main__":
    main()