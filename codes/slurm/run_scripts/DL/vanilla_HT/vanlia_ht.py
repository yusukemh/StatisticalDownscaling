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

# define a NN model
def define_model(
    input_dim=20, 
    lr=0.005, 
    activation='relu',
    n_units=256,
    n_layers=4,
    dropout=0.5
):
    inputs = Input(shape=(input_dim,))
    x = Dense(units=n_units, activation=activation)(inputs)
    
    for i in range(n_layers - 1):
        if dropout:
            x = Dropout(rate=dropout)(x)
        x = Dense(units=n_units, activation=activation)(x)
    outputs = Dense(units=1, kernel_initializer='normal', activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=[RootMeanSquaredError()]
    )
    
    return model

def main():
    file_name = './progress.txt'
    with open(file_name, 'w') as f:
        f.write('')# clear the content

    columns = C_SINGLE
    # load nonfilled dataset
    df_nonfilled = pd.read_csv(f"{BASE_DIR}/nonfilled_dataset.csv", usecols=C_SINGLE + C_COMMON)
    # sample a station: returned object is sorted.
    df_station = sample_station(df=df_nonfilled, threshold=750, seed=42)

    parameters = [
        sherpa.Continuous(name='lr', range=[0.001, 0.01]),
        sherpa.Choice(name='activation', range=['relu', 'elu', 'tanh']),
        sherpa.Discrete(name='n_units', range=[128, 1024]),
        sherpa.Discrete(name='n_layers', range=[1,5]),
        sherpa.Choice(name='dropout', range=[0.3, 0.8]),
        sherpa.Discrete(name='batch_size', range=[32, 128])    
    ]
    
    num_trials = 5
    alg = sherpa.algorithms.RandomSearch(max_num_trials=num_trials)
    study = sherpa.Study(
        parameters=parameters,
        algorithm=alg,
        lower_is_better=True
    )
    
    
    for _, trial in enumerate(study):
        with open(file_name, 'a') as f:
            f.write(f"trial {_}/{num_trials}\n")
        start = time.time()
        model_params = {
            "input_dim": len(columns),
            "lr": trial.parameters['lr'],
            "activation": trial.parameters['activation'],
            "n_units": trial.parameters['n_units'],
            "n_layers": trial.parameters['n_layers'],
            "dropout": trial.parameters['dropout'],
        }
        batch_size = trial.parameters['batch_size']

        # first, estimate the number of epochs
        X = np.array(df_station[C_SINGLE])
        Y = np.array(df_station['data_in'])
        estimated_epochs = estimate_epochs(
            X=X, Y=Y, model_func=define_model,
            model_params=model_params,
            patience=5,
            n_iter=10,
            add_noise=False
        )

        # now evaluate the performance, use 10 iterations
        for iteration in range(10):
            y_pred = cross_val_predict_for_nn(
                model_func=define_model,
                model_params=model_params,
                batch_size=batch_size,
                epochs=int(estimated_epochs),
                X=X, Y=Y,
                callback=None,
                early_stopping=False,
                val_size=0, # ignored, as early_stoppinf=False
                verbose=False,
                add_noise=False
            )

            validation_error = mean_squared_error(Y, y_pred, squared=False)
            study.add_observation(
                trial=trial,
                iteration=iteration,
                objective=validation_error,
                context={'validation_error': validation_error}
            )
        study.finalize(trial)
        end = time.time()
        elapsed_time = int(end - start)
        # print(f'Elapsed time = {elapsed_time}')
        with open(file_name, 'a') as f:
            f.write(f'Elapsed time = {elapsed_time}\n')
            
        study.save(f'./') # save the result
    
if __name__ == '__main__':
    main()