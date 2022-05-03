# CHECK FILE NAME, MODEL, PARAMETER SPACE
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

# Variables from config file
# from config import BASE_DIR, FILE_NAMES, LABELS, ATTRIBUTES, BEST_MODEL_COLUMNS, ISLAND_RANGES

BASE_DIR='/home/yusukemh/github/yusukemh/StatisticalDownscaling/dataset'

def cross_val_predict_for_nn(model, X, Y, callback, batch_size, epochs, verbose):
    kf = KFold(n_splits=5)
    y_pred = []

    for train_index, test_index in kf.split(X):
        Xtemp, Xtest = X[train_index], X[test_index]
        Ytemp, Ytest = Y[train_index], Y[test_index]
        
        Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(Xtemp, Ytemp, test_size=0.2, shuffle=True)
        # print(Xtrain.shape, Xvalid.shape, Xtest.shape)
        
        # scale the input
        scaler = StandardScaler()
        Xtrain = scaler.fit_transform(Xtrain)
        Xvalid = scaler.transform(Xvalid)
        Xtest = scaler.transform(Xtest)
        
        model.fit(
            Xtrain, Ytrain, epochs=epochs,
            validation_data = (Xvalid, Yvalid),
            callbacks=[callback],
            batch_size=batch_size,
            verbose=verbose
        )
        y_pred.extend(model.predict(Xtest).tolist())
    
    return np.array(y_pred)

def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def define_model(num_inputs=20, lr=0.0065):
    inputs = Input(shape=(num_inputs,))
    x = Dense(units=20, activation='relu')(inputs)
    # x = Dense(units=16, activation='relu')(inputs)
    x = Dense(units=8, activation='relu')(x)
    x = Dense(units=4, activation='relu')(x)
    outputs = Dense(units=1, kernel_initializer='normal')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=[RootMeanSquaredError()]
    )
    
    return model

def main():
    file_name = './result_6grids.txt'

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

    columns.extend(['lat', 'lon', 'elevation', 'season_wet', 'season_dry'])
    
    # load datasets
    df_train = pd.read_csv(f"{BASE_DIR}/train.csv")
    df_valid = pd.read_csv(f"{BASE_DIR}/valid.csv")
    df_test = pd.read_csv(f"{BASE_DIR}/test.csv")

    df_combined = pd.concat([df_train, df_valid, df_test])
    
    results = []
    n_stations = df_combined['skn'].unique().shape[0]
    lr = 0.003
    for i, skn in enumerate(df_combined['skn'].unique()[:10]):
        df_station = df_combined[df_combined['skn'] == skn]
        if df_station.shape[0] <= 5: continue

        df_station = df_station.sort_values(by=['year', 'month'])

        X = np.array(df_station[columns])
        Y = np.array(df_station['data_in'])

        model = define_model(lr=lr, num_inputs=len(columns))
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 3, mode='min')
        epochs=20
        batch_size=64

        yhat_nn = cross_val_predict_for_nn(model, X, Y, callback, batch_size, epochs, verbose=0)

        model = LinearRegression()
        yhat_lr = cross_val_predict(model, X, Y)

        results.append(
            pd.DataFrame(
                {
                    "skn": [skn for _ in range(X.shape[0])],
                    "data_in": Y,
                    "pred_nn": yhat_nn.reshape(-1,),
                    "pred_lr": yhat_lr
                }
            )
        )
        
        line = f"{i}/{n_stations}"
        print(line, end='\r')

        with open(file_name, 'a') as f:
            f.write(line)
            
    pd.concat(results).to_csv(f"./result_nn_6grids.csv", index=False)
if __name__ == "__main__":
    main()