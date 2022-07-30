import sherpa
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
sys.path.append('/home/yusukemh/github/yusukemh/StatisticalDownscaling/writeup/')
from config import C_COMMON, C_GRID, C_SINGLE, FILENAME
from sklearn.model_selection import train_test_split
import time

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

columns = C_SINGLE
column_type = 'single'

def assign_inner_fold(df, n_folds=5):
    # assign fold for each sample
    df_len_by_month = pd.DataFrame(df.groupby(by=['year', 'month']).size()).reset_index().rename({0: "len"}, axis=1)
    df_len_by_month = df_len_by_month.sort_values(['year', 'month'])
    df_len_by_month['cumsum'] = df_len_by_month['len'].cumsum()
    n_samples_total = df_len_by_month['cumsum'].iloc[-1]
    n_samples_per_fold = np.ceil(n_samples_total / n_folds)
    
    df_len_by_month['inner_fold'] = df_len_by_month.apply(lambda row: int(row['cumsum'] / n_samples_per_fold), axis=1)
    
    df_w_fold = pd.merge(left=df, right=df_len_by_month, left_on=['year', 'month'], right_on=['year', 'month'])
    
    return df_w_fold

def define_model(
    input_dim=20,
    n_units=512,
    activation='selu',#selu
    learning_rate=0.00001,
    loss='mse'
):
    inputs = Input(shape=(input_dim))
    x = Dense(units=n_units, activation=activation)(inputs)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=n_units, activation=activation)(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=n_units, activation=activation)(x)
    x = Dropout(rate=0.5)(x)# serves as regularization
    # x = Dense(units=int(n_units/2), activation=activation)(x)
    outputs = Dense(units=1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[RootMeanSquaredError()]
    )
    return model

def prepare_dataset(df, skn, inner_fold):
    """
    Splits dataset into train and test, and scales x
    """
    df_station = df[df['skn'] == skn]
    df_train = df_station[df_station['inner_fold'] != inner_fold]
    df_test = df_station[df_station['inner_fold'] == inner_fold]
    x_train, x_test = np.array(df_train[columns]), np.array(df_test[columns])
    y_train, y_test = np.array(df_train['data_in']), np.array(df_test['data_in'])
    
    x_scaler = MinMaxScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)
    
    return x_train, x_test, y_train, y_test

def transform_y(y_train, y_test):
    scaler = MinMaxScaler(feature_range=(0,1))
    y_train = np.log(y_train + 1.)
    y_test = np.log(y_test + 1.)
    
    y_train = scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler.transform(y_test.reshape(-1, 1))
    
    return y_train, y_test, scaler
    
def inverse_transform_y(y, scaler):
    y = scaler.inverse_transform(y)
    y = np.power(np.e, y) - 1
    return y
    

def main():
    columns = C_SINGLE
    column_type = 'single'
    df = pd.read_csv(FILENAME, usecols=C_COMMON + columns).sort_values(['year', 'month'])

    # we only need the training folds
    df_train = df.query('fold != 4')
    # assert (sorted(df_test_outer['skn'].unique()) == sorted(df_train_outer['skn'].unique()))
    df_train = assign_inner_fold(df_train)
    
    parameters = [
        sherpa.Continuous('n_units', [256, 512, 1024]),
        sherpa.Continuous('learning_rate', [0.00001, 0.01]),
        sherpa.Choice('batch_size', [64, 128, 192, 256, 512]),
        sherpa.Choice('loss', ['mse', 'mae'])
    ]
    n_run = 240
    alg = sherpa.algorithms.RandomSearch(max_num_trials=n_run)
    study = sherpa.Study(parameters=parameters, algorithm=alg, lower_is_better=True)
    dfs = []

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=20,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.95,
            patience=10
        )
    ]
    dfs = []
    for i, trial in enumerate(study):
        start = time.time()
        params = {
            'input_dim': len(columns),
            'n_units': trial.parameters['n_units'],
            'learning_rate': trial.parameters['learning_rate'],
            'loss': trial.parameters['loss']
        }
        line = '===============================================\n'
        line += str(params) + '\n'
        
        batch_size = trial.parameters['batch_size']

        for skn in tqdm(df_train['skn'].unique()):        
            ytest_station = []
            yhat_station = []
            for inner_fold in range(5):
                x_train, x_test, y_train, y_test = prepare_dataset(df_train, skn=skn, inner_fold=inner_fold)
                y_train, y_test, scaler = transform_y(y_train, y_test)
                model = define_model(**params)
                model.fit(x_train, y_train, epochs=500, validation_split=0.2, callbacks=callbacks,
                    batch_size=batch_size,
                    verbose=0
                )
                yhat = model.predict(x_test)
                yhat = inverse_transform_y(yhat, scaler)
                y_test = inverse_transform_y(y_test, scaler)

                # record the result
                yhat_station.extend(yhat)
                ytest_station.extend(y_test)

            mae_station = mean_absolute_error(ytest_station, yhat_station)
            rmse_station = mean_squared_error(ytest_station, yhat_station, squared=False)

            _ = pd.DataFrame([params])
            _['skn'] = [skn]
            _['batch_size'] = [batch_size]
            _['mae'] = [mae_station]
            _['rmse'] = [rmse_station]
            _['trial_id'] = [i]
            dfs.append(_)
        end = time.time()
        
        pd.concat(dfs).to_csv(f'nn_report_{n_run}_{column_type}.csv')
        line += "elapsed time         : {:.3f}".format(end - start) + '\n'
        with open('progress.txt', 'a') as f:
            f.write(line)

if __name__ == '__main__':
    main()