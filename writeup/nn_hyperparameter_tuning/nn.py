import sherpa
import pandas as pd
import numpy as np
from tqdm import tqdm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

import sys
sys.path.append('/home/yusukemh/github/yusukemh/StatisticalDownscaling/writeup')
from config import C_COMMON, C_GRID, C_SINGLE, FILENAME
from util import load_data, NeuralNetwork

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2s

def define_model(
    input_dim=20,
    n_units=512,
    activation='selu',#selu
    learning_rate=0.00001,
    loss='mse',
    batch_size=64
):
    inputs = Input(shape=(input_dim))
    x = Dense(units=n_units, activation=activation, kernel_regularizer=L2(l2=0.01))(inputs)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=n_units, activation=activation, kernel_regularizer=L2(l2=0.01))(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=n_units, activation=activation, kernel_regularizer=L2(l2=0.01))(x)
    x = Dropout(rate=0.5)(x)# serves as regularization
    outputs = Dense(units=1, activation='softplus')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[RootMeanSquaredError()]
    )
    return model, batch_size


def main():
    columns, col_type = C_SINGLE, 'single'
    n_run = 500
    # load data
    df_train, _ = load_data(columns + C_COMMON, FILENAME)
    
    # set up sherpa
    #====================================================================
    parameters = [
        sherpa.Discrete('n_units', [256, 1024]),
        sherpa.Continuous('learning_rate', [0.00001, 0.01]),
        sherpa.Choice('batch_size', [64, 128, 192, 256, 512, 1024]),
        sherpa.Choice('loss', ['mse', 'mae'])
    ]
    study = sherpa.Study(
        parameters=parameters,
        algorithm=sherpa.algorithms.RandomSearch(max_num_trials=n_run),
        lower_is_better=True
    )
    #====================================================================
    # run experiment
    dfs = []
    for i, trial in enumerate(study):
        start = time.time()
        # obtain hyperparameters
        params = {key: val for key, val in trial.parameters.items()}
        params['input_dim'] = len(columns)
        line = '===============================================\n'
        line += str(params) + '\n'

        for skn in tqdm(df_train['skn'].unique()):
            model = NeuralNetwork(
                model_func=define_model,
                params=params,
                columns=columns
            )
            ret = model.cross_val_predict(df_train, skn)
            df = pd.DataFrame()
            df['trial_id'] = i,
            df['rmse'] = [ret['rmse']]
            df['mae'] = [ret['mae']]
            df['epochs'] = [ret['epochs']]
            df['skn'] = [skn]
            for key, val in params.items():
                df[key] = [val]
            dfs.append(df)
        pd.concat(dfs).to_csv(f'nn_report_{n_run}_{col_type}.csv')
        end = time.time()
        line += "elapsed time         : {:.3f}".format(end - start) + '\n'
        with open('progress.txt', 'a') as f:
            f.write(line)

if __name__ == '__main__':
    main()