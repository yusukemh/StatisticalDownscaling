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

def main():
    # Variables from config file
    BASE_DIR = "/home/yusukemh/github/yusukemh/StatisticalDownscaling/dataset"
    df_6grids = pd.read_csv(f"{BASE_DIR}/dataset_5girds.csv")
    df_6g = df_6grids.drop(
        labels=["skn", "year", "month", "name", "Observer", "NumMos", "MinYear", "MaxYear", "Status2010"],
        axis=1
    )
    df_6g = df_6g.drop(
        labels=[
            "air2m", "air1000_500", "hgt500", "hgt1000", "omega500",
            "pottemp1000-500", "pottemp1000-850", "pr_wtr", "shum-uwnd-700",
            "shum-uwnd-925", "shum-vwnd-700", "shum-vwnd-950", "shum700",
            "shum925", "skt", "slp"
        ],
        axis=1
    )
    
    Y = df_6g["data_in"]
    X = df_6g.drop(["data_in"], axis=1)

    Xtemp, Xtest, Ytemp, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
    Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(Xtemp, Ytemp, test_size=0.25, random_state=42)
    
    gradient_boost = GradientBoostingRegressor(
        n_estimators=280, 
        learning_rate=0.5,
        max_depth=5,
        min_samples_split=2,
        verbose=False
    )

    gradient_boost.fit(Xtrain, Ytrain)
    print("MSE on gradient boost (test) : {:.5f}".format(mean_squared_error(Ytest, gradient_boost.predict(Xtest))))
    print("MSE on gradient boost (train): {:.5f}".format(mean_squared_error(Ytrain, gradient_boost.predict(Xtrain))))
    
    file_name = './XGB_GBR_6grid.txt'
    
    line = "Result on gradient boosting regressor\n"
    line += "MSE on gradient boost (test) : {:.5f}".format(mean_squared_error(Ytest, gradient_boost.predict(Xtest))) + '\n'
    line += "MSE on gradient boost (train): {:.5f}".format(mean_squared_error(Ytrain, gradient_boost.predict(Xtrain))) + '\n'
    
    
    xgboost = XGBRegressor(
        n_estimators=110,
        learning_rate=0.5,
        max_depth=9,
        verbosity=0
    )
    xgboost.fit(Xtrain, Ytrain)
    print("MSE on xgboost (test) : {:.5f}".format(mean_squared_error(Ytest, xgboost.predict(Xtest))))
    print("MSE on xgboost (train): {:.5f}".format(mean_squared_error(Ytrain, xgboost.predict(Xtrain))))
    
    line += "Result on xgboost regressor\n"
    line += "MSE on xgboost (test) : {:.5f}".format(mean_squared_error(Ytest, xgboost.predict(Xtest))) + '\n'
    line += "MSE on xgboost (train): {:.5f}".format(mean_squared_error(Ytrain, xgboost.predict(Xtrain))) + '\n'
    
    with open(file_name, 'a') as f:
        f.write(line)

if __name__ == "__main__":
    main()