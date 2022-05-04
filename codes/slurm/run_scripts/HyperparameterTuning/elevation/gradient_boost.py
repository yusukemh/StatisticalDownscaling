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
LABELS=[
    'air2m', 'air1000_500', 'hgt500', 'hgt1000', 'omega500',
    'pottemp1000-500', 'pottemp1000-850', 'pr_wtr', 'shum-uwnd-700',
    'shum-uwnd-925', 'shum-vwnd-700', 'shum-vwnd-950', 'shum700', 'shum925', 'skt', 'slp'
]

base_columns = ["data_in", "lat", "lon", "elevation"]

def main():
    # load datasets
    df_train = pd.read_csv(f"{BASE_DIR}/train.csv")
    df_valid = pd.read_csv(f"{BASE_DIR}/valid.csv")
    df_test = pd.read_csv(f"{BASE_DIR}/test.csv")
    
    columns = (base_columns + LABELS)
    line = f"labels to use: {columns}\n"

    Xtrain = np.array(df_train[columns].drop(labels=["data_in"], axis=1))
    Ytrain = np.array(df_train["data_in"])

    Xvalid = np.array(df_valid[columns].drop(labels=["data_in"], axis=1))
    Yvalid = np.array(df_valid["data_in"])

    Xtest = np.array(df_test[columns].drop(labels=["data_in"], axis=1))
    Ytest = np.array(df_test["data_in"])
    
   
    
    file_name = './GBR_elevation.txt'

    parameters = [
        sherpa.Choice('n_estimators', list(range(100, 310, 10))),
        sherpa.Choice('learning_rate', [0.05, 0.1, 0.5, 1.0, 1.25, 1.5, 2]),
        sherpa.Discrete('max_depth', [1, 8]),
        sherpa.Discrete('min_samples_split', [2, 10])
    ]

    alg = sherpa.algorithms.RandomSearch(max_num_trials=50)
    study = sherpa.Study(parameters=parameters,
                         algorithm=alg,
                         lower_is_better=True)

    for trial in study:
        start = time.time()
        line += '===============================================\n'
        params = {
            "n_estimators": trial.parameters['n_estimators'],
            "learning_rate": trial.parameters['learning_rate'],
            "max_depth": trial.parameters['max_depth'],
            "min_samples_split": trial.parameters["min_samples_split"],
            "verbose": True
        }
        print(params)
        line += str(params) + '\n'
        model = GradientBoostingRegressor(**params)
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