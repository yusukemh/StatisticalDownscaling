# basic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#tensorflow
import tensorflow as tf
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout

import tensorflow_probability as tfp
tfd = tfp.distributions

# others
from copy import deepcopy
from xgboost import XGBRegressor
import sherpa
import sys

# Variables from config file
sys.path.append('/home/yusukemh/github/yusukemh/StatisticalDownscaling/codes/')
from config import BASE_DIR, FILE_NAMES, LABELS, ATTRIBUTES, BEST_MODEL_COLUMNS, ISLAND_RANGES, C_SINGLE, C_INT50, C_INT100, C_GRID, C_COMMON

# util
from util import cross_val_predict_for_nn, estimate_epochs, define_model, define_hetero_model_normal, define_hetero_model_gamma

def run_single_experiment(
    X, Y,
    model_func, model_params, batch_size,
    n_trial, skn
):
    # first, run linear regression
    linear_regression = LinearRegression()
    y_pred = cross_val_predict(linear_regression, X, Y, n_jobs=-1)
    rmse_lr = mean_squared_error(Y, y_pred, squared=False)
    # estimate the # epochs
    estimated_epochs = estimate_epochs(
        X=X, Y=Y, model_func=model_func, model_params=model_params, batch_size=batch_size, n_iter=30,
    )
    
    rmses = []
    for trial in range(n_trial):
        y_pred = cross_val_predict_for_nn(
            X=X, Y=Y, model_func=model_func, model_params=model_params, callback=None, batch_size=64,
            epochs=int(estimated_epochs), early_stopping=False, verbose=False
        )
        rmse = mean_squared_error(Y, y_pred, squared=False)
        rmses.append(rmse)
    rmses = np.array(rmses)
    m, s = np.mean(rmses), np.std(rmses)
    return pd.DataFrame(
        dict(
            n_samples=X.shape[0],
            estimated_epochs=estimated_epochs,
            rmse_LR=rmse_lr,
            rmse_NN_mean=m,
            rmse_NN_std=s,
            rel_imp=(rmse_lr - m)/rmse_lr
        ),
        index=[skn]
    )

def main():
    file_name = './progress.txt'

    columns = C_SINGLE
    # load nonfilled dataset
    df_nonfilled = pd.read_csv(f"{BASE_DIR}/nonfilled_dataset.csv", usecols=C_SINGLE + C_COMMON)
    # sample a station: returned object is sorted.

    df_n_data = df_nonfilled.groupby('skn').size().reset_index().rename(columns={0:"n_data"})
    valid_skn = df_n_data[df_n_data['n_data'] > 750]['skn']

    stats_regular = []
    stats_normal = []
    stats_gamma = []

    for i, skn in enumerate(valid_skn):
        with open(file_name, 'a') as f:
            f.write(f'Running experiment on skn {skn}\n')
            f.write(f'{i}/{valid_skn.shape[0]}\n')

        df_station = df_nonfilled[df_nonfilled['skn'] == skn].sort_values(['year', 'month'])

        X = np.array(df_station[columns])
        Y = np.array(df_station['data_in'])

        with open(file_name, 'a') as f:
            f.write(f'\tRunning regular NN\n')

        # regular NN
        #############################################
        model_params = {
            "input_dim"         : len(columns),
            "activation"        : 'relu',
            "dropout"           : 0.4578657205946084,
            "lr"                : 0.0043961731420066,
            "n_layers"          : 2,
            "n_units"           : 905,
        }
        batch_size = 116
        #############################################
        stats_regular.append(
            run_single_experiment(
                X, Y, model_func=define_model, model_params=model_params,
                batch_size=batch_size, n_trial=20, skn=skn
            )
        )

        with open(file_name, 'a') as f:
            f.write(f'\tRunning normal NN\n')

        #############################################            
        model_params = {
            "input_dim"         : len(columns),
            "activation"        : 'relu',
            "dropout"           : 0.561853280248839,
            "l2_sigma"          : 240,
            "lr"                : 0.0085826439625522,
            "n_additional_layers_main": 1,
            "n_additional_layers_sub": 1,
            "n_units_main"      : 372,
            "n_units_sub"       : 260,
            "sigma_a"           : 0.0071237007950889,
            "sigma_b"           : 0.0207931459510849,
        }
        batch_size = 117
        #############################################
        stats_normal.append(
            run_single_experiment(
                X, Y, model_func=define_hetero_model_normal, model_params=model_params,
                batch_size=batch_size, n_trial=20, skn=skn
            )
        )

        with open(file_name, 'a') as f:
            f.write(f'\tRunning gamma NN\n')

        ############################################# 
        model_params = {
            "input_dim"         : len(columns),
            "activation"        : 'relu',
            "dropout"           : 0.7006435973994982,
            "lr"                : 0.0007538441168978,
            "n_additional_layers_main": 0,
            "n_additional_layers_sub": 0,
            "n_units_main"      : 399,
            "n_units_sub"       : 400,
        }
        batch_size = 60
        #############################################
        stats_gamma.append(
            run_single_experiment(
                X, Y, model_func=define_hetero_model_gamma, model_params=model_params,
                batch_size=batch_size, n_trial=20, skn=skn
            )
        )

    pd.concat(stats_regular).to_csv(f"./stats_regular.csv", index=False)
    pd.concat(stats_normal).to_csv(f"./stats_normal.csv", index=False)
    pd.concat(stats_gamma).to_csv(f"./stats_gamma.csv", index=False)

if __name__ == '__main__':
    main()