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
from util import cross_val_predict_for_nn, estimate_epochs

# define models
def define_model(input_dim=20, lr=0.005):
    inputs = Input(shape=(input_dim,))
    x = Dense(units=256, activation='elu')(inputs)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=256, activation='elu')(x)
    outputs = Dense(units=1, kernel_initializer='normal', activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=[RootMeanSquaredError()]
    )
    
    return model

def define_hetero_model_normal(input_dim=20, lr=0.0065):
    inputs = Input(shape=(input_dim,))
    x = Dense(units=512, activation='selu', kernel_initializer='normal')(inputs)
    x = Dense(units=512, activation='selu', kernel_initializer='normal')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=512, activation='selu', kernel_initializer='normal')(x)
    x = Dropout(rate=0.5)(x)
    
    m = Dense(units=256, activation='selu', kernel_initializer='normal')(x)
    m = Dense(units=10, activation='selu', kernel_initializer='normal')(m)
    m = Dense(units=1, activation='linear', kernel_initializer='normal')(m)
    
    s = Dense(units=256, activation='selu', kernel_initializer='normal')(x)
    s = Dense(units=10, activation='selu', kernel_initializer='normal')(s)
    s = Dense(units=1, activation='linear', kernel_initializer='normal', kernel_regularizer=tf.keras.regularizers.L2(l2=100))(s)
    
    ms = Concatenate(axis=-1)([m, s])
    outputs = tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: tfd.Normal(
            loc=2.5 * t[...,0] + 0.01, scale=tf.math.softplus(0.001*t[...,1]+0.03)#this part is important
        ),
        convert_to_tensor_fn=lambda s: s.mean()
    )(ms)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=lr),
        loss=lambda y, p_y: -p_y.log_prob(y),
        metrics=[RootMeanSquaredError()]
    )
    
    return model

def safe_nll(y, p_y):
    epsilon=1e-5
    return -p_y.log_prob(y + epsilon) # y = 0 yields nan

def define_hetero_model_gamma(input_dim=20, lr=0.0065):
    inputs = Input(shape=(input_dim,))
    x = Dense(units=512, activation='elu', kernel_initializer='normal')(inputs)
    x = Dense(units=512, activation='elu', kernel_initializer='normal')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=512, activation='elu', kernel_initializer='normal')(x)
    x = Dropout(rate=0.5)(x)
    
    m = Dense(units=256, activation='elu', kernel_initializer='normal')(x)
    m = Dense(units=10, activation='elu', kernel_initializer='normal')(m)
    m = Dense(units=1, activation='linear', kernel_initializer='normal')(m)
    
    s = Dense(units=256, activation='elu', kernel_initializer='normal')(x)
    s = Dense(units=10, activation='elu', kernel_initializer='normal')(s)
    s = Dense(units=1, activation='linear', kernel_initializer='normal')(s)
    
    ms = Concatenate(axis=-1)([m, s])
    outputs = tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: tfd.Gamma(
            concentration=tf.math.softplus(t[...,0]), rate=tf.math.softplus(t[...,1])
        ),
        convert_to_tensor_fn=lambda d: d.mean()
    )(ms)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=lr),
        #loss=lambda y, p_y: -p_y.log_prob(y),
        loss=safe_nll,
        metrics=[RootMeanSquaredError()]
    )
    
    return model

def run_single_experiment(
    X, Y,
    model_func, model_params,
    n_trial
):
    # first, run linear regression
    linear_regression = LinearRegression()
    y_pred = cross_val_predict(linear_regression, X, Y, n_jobs=-1)
    rmse_lr = mean_squared_error(Y, y_pred, squared=False)
    # estimate the # epochs
    estimated_epochs = estimate_epochs(
        X=X, Y=Y, model_func=model_func, model_params=model_params, n_iter=50
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
        stats_regular.append(
            run_single_experiment(
                X, Y, model_func=define_model, model_params=dict(input_dim=len(columns), lr=0.0005),
                n_trial=30
            )
        )

        with open(file_name, 'a') as f:
            f.write(f'\tRunning normal NN\n')

        stats_normal.append(
            run_single_experiment(
                X, Y, model_func=define_hetero_model_normal, model_params=dict(input_dim=len(columns), lr=0.0005),
                n_trial=30
            )
        )

        with open(file_name, 'a') as f:
            f.write(f'\tRunning gamma NN\n')

        stats_gamma.append(
            run_single_experiment(
                X, Y, model_func=define_hetero_model_gamma, model_params=dict(input_dim=len(columns), lr=0.001),
                n_trial=30
            )
        )

    pd.concat(stats_regular).to_csv(f"./stats_regular.csv", index=False)
    pd.concat(stats_normal).to_csv(f"./stats_normal.csv", index=False)
    pd.concat(stats_gamma).to_csv(f"./stats_gamma.csv", index=False)

if __name__ == '__main__':
    main()