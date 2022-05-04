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
    file_name = './XGB_interp100.txt'

    columns = [
        'data_in',
        'lat', 'lon', 'elevation', 'season_wet', 'season_dry',
        'i100_air2m', 'i100_air1000_500', 'i100_hgt500', 'i100_hgt1000', 'i100_omega500',
        'i100_pottemp1000-500', 'i100_pottemp1000-850', 'i100_pr_wtr', 'i100_shum-uwnd-700',
        'i100_shum-uwnd-925', 'i100_shum-vwnd-700', 'i100_shum-vwnd-950', 'i100_shum700', 'i100_shum925', 
        'i100_skt', 'i100_slp'
    ]
    
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
    
    parameters = [
        sherpa.Choice('n_estimators', list(range(100, 310, 10))),
        sherpa.Choice('learning_rate', [0.05, 0.1, 0.5, 1.0, 1.25, 1.5, 2]),
        sherpa.Discrete('max_depth', [1, 10]),
    ]
    

    alg = sherpa.algorithms.RandomSearch(max_num_trials=50)
    study = sherpa.Study(parameters=parameters,
                         algorithm=alg,
                         lower_is_better=True)

    for trial in study:
        start = time.time()
        line = '===============================================\n'
        params = {
            "n_estimators": trial.parameters['n_estimators'],
            "learning_rate": trial.parameters['learning_rate'],
            "max_depth": trial.parameters['max_depth'],
            "verbosity": 1
        }
        print(params)
        line += str(params) + '\n'
        model = XGBRegressor(**params)
        model.fit(Xtrain, Ytrain)
        training_error = mean_squared_error(Ytrain, model.predict(Xtrain))
        validation_error = mean_squared_error(Yvalid, model.predict(Xvalid))
        study.add_observation(
            trial=trial,
            iteration=1,
            objective=validation_error,
            context={'training_error': training_error}
        )
        end = time.time()
        line += "MSE on training set  : {:.6f}".format(training_error) + '\n'
        line += "MSE on validation set: {:.6f}".format(validation_error) + '\n'
        line += "elapsed time         : {:.3f}".format(end - start) + '\n'

        with open(file_name, 'a') as f:
            f.write(line)

        study.finalize(trial)

    print(study.get_best_result())
if __name__ == "__main__":
    main()