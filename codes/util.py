import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from tensorflow.keras.callbacks import EarlyStopping
#tensorflow
import tensorflow as tf
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout

import tensorflow_probability as tfp
tfd = tfp.distributions


# from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
from copy import deepcopy

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
    outputs = Dense(units=1, kernel_initializer=tf.keras.initializers.HeNormal, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=[RootMeanSquaredError()]
    )
    
    return model

def define_hetero_model_gamma(
    input_dim=20, lr=0.0065,
    n_additional_layers_main=2,
    n_additional_layers_sub=0,
    activation='elu',
    n_units_main=512,
    n_units_sub=256,
    dropout=0.5,
):
    inputs = Input(shape=(input_dim,))
    x = Dense(units=n_units_main, activation=activation, kernel_initializer=tf.keras.initializers.HeNormal)(inputs)
    
    for _ in range(n_additional_layers_main):
        x = Dense(units=n_units_main, activation=activation, kernel_initializer=tf.keras.initializers.HeNormal)(x)
        if dropout:
            x = Dropout(rate=0.5)(x)
    
    m = Dense(units=n_units_sub, activation=activation, kernel_initializer=tf.keras.initializers.HeNormal)(x)
    for _ in range(n_additional_layers_sub):
        m = Dense(units=n_units_sub, activation=activation, kernel_initializer=tf.keras.initializers.HeNormal)(m)
    m = Dense(units=10, activation=activation, kernel_initializer=tf.keras.initializers.HeNormal)(m)
    m = Dense(units=1, activation='linear', kernel_initializer=tf.keras.initializers.HeNormal)(m)
    
    s = Dense(units=n_units_sub, activation=activation, kernel_initializer=tf.keras.initializers.HeNormal)(x)
    for _ in range(n_additional_layers_sub):
        s = Dense(units=n_units_sub, activation=activation, kernel_initializer=tf.keras.initializers.HeNormal)(s)
    s = Dense(units=10, activation=activation, kernel_initializer=tf.keras.initializers.HeNormal)(s)
    s = Dense(units=1, activation='linear', kernel_initializer=tf.keras.initializers.HeNormal)(s)
    
    ms = Concatenate(axis=-1)([m, s])
    outputs = tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: tfd.Gamma(
            concentration=tf.math.softplus(t[...,0]), rate=tf.math.softplus(t[...,1])
        ),
        convert_to_tensor_fn=lambda d: d.mean()
    )(ms)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    epsilon=1e-5 # for loss function
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=lr),
        loss=lambda y, p_y: -p_y.log_prob(y + epsilon),
        # loss=safe_nll,
        metrics=[RootMeanSquaredError()]
    )
    
    return model

@tf.autograph.experimental.do_not_convert
def define_hetero_model_normal(
    input_dim, lr=0.0065,
    n_additional_layers_main=0,
    n_additional_layers_sub=0,
    activation='relu',
    n_units_main=512,
    n_units_sub=10,
    dropout=0.5,
    l2_sigma=100,
    sigma_a=0.001,# these hyperparameters might have significant affect
    sigma_b=0.03 # these hyperparameters might have signigicant affect
):
    inputs = Input(shape=(input_dim,))
    x = Dense(units=n_units_main, activation=activation, kernel_initializer=tf.keras.initializers.HeNormal)(inputs)
    # main branch
    for _ in range(n_additional_layers_main):
        if dropout:
            x = Dropout(rate=dropout)(x)
        x = Dense(units=n_units_main, activation=activation, kernel_initializer=tf.keras.initializers.HeNormal)(x)
    # mean branch
    if n_additional_layers_sub == 0:
        m = Dense(units=1, activation='linear', kernel_initializer=tf.keras.initializers.HeNormal)(x)
    else:
        m = Dense(units=n_units_sub, activation=activation, kernel_initializer=tf.keras.initializers.HeNormal)(x)
        for _ in range(n_additional_layers_sub - 1):
            m = Dense(units=n_units_sub, activation=activation)(m)
        m = Dense(units=1, activation='linear', kernel_initializer=tf.keras.initializers.HeNormal)(m)
    
    # std branch
    if n_additional_layers_sub == 0:
        s = Dense(units=1, activation='linear', kernel_initializer=tf.keras.initializers.HeNormal, kernel_regularizer=tf.keras.regularizers.L2(l2=l2_sigma))(x)
    else:
        s = Dense(units=n_units_sub, activation=activation, kernel_initializer=tf.keras.initializers.HeNormal)(x)
        for _ in range(n_additional_layers_sub - 1):
            s = Dense(units=n_units_sub, activation=activation)(s)
        s = Dense(units=1, activation='linear', kernel_initializer=tf.keras.initializers.HeNormal, kernel_regularizer=tf.keras.regularizers.L2(l2=l2_sigma))(s)
    
    ms = Concatenate(axis=-1)([m, s])
    outputs = tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: tfd.Normal(
            loc=t[...,0], scale=tf.math.softplus(sigma_a * t[...,1] + sigma_b)
        ),
        convert_to_tensor_fn=lambda s: s.mean()
    )(ms)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=lr),
        # loss=lambda y, p_y: -p_y.log_prob(y),
        loss=lambda y, p_y: -p_y.log_prob(y + 1e-5),
        metrics=[RootMeanSquaredError()]
    )
    
    return model

def augment_data(X, Y):
    # make sure to apply this only on the training dataset
    # add noise to X
    new_X = [X]
    new_Y = [Y]
    for _ in range(10):
        noise = np.random.random(X.shape) * 0.001
        new_X.append(X + noise)
        new_Y.append(Y)
        
    
    return (np.vstack(new_X), np.array(new_Y).flatten())

def cross_val_predict_for_nn(
    model_func, model_params, X, Y,
    callback, batch_size, epochs,
    early_stopping=True,
    val_size=0.2,
    add_noise=False,
    verbose=False
):
    kf = KFold(n_splits=5)
    y_pred = []
    
    for train_index, test_index in kf.split(X):
        model = model_func(**model_params)
        if early_stopping:
            Xtemp, Xtest = X[train_index], X[test_index]
            Ytemp, Ytest = Y[train_index], Y[test_index]

            Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(Xtemp, Ytemp, test_size=val_size, shuffle=True)

            # scale the input
            scaler = StandardScaler()
            Xtrain = scaler.fit_transform(Xtrain)
            Xvalid = scaler.transform(Xvalid)
            Xtest = scaler.transform(Xtest)
            # if early_stopping is true, then callback must not be None
            model.fit(
                Xtrain, Ytrain, epochs=epochs,
                validation_data = (Xvalid, Yvalid),
                callbacks=[callback],
                batch_size=batch_size,
                verbose=verbose
            )
            y_pred.extend(model.predict(Xtest).tolist())
        else: #if no early stopping
            Xtrain, Xtest = X[train_index], X[test_index]
            Ytrain, Ytest = Y[train_index], Y[test_index]
            if add_noise:
                Xtrain, Ytrain = augment_data(Xtrain, Ytrain)
            
            scaler = StandardScaler()
            Xtrain = scaler.fit_transform(Xtrain)
            Xtest = scaler.transform(Xtest)
            if callback is None:
                model.fit(
                    Xtrain, Ytrain, epochs=epochs,
                    batch_size=batch_size,
                    verbose=verbose
                )
            else:
                model.fit(
                    Xtrain, Ytrain, epochs=epochs,
                    callbacks=[callback],
                    batch_size=batch_size,
                    verbose=verbose
                )
            y_pred.extend(model.predict(Xtest).tolist())
            
            
    
    return np.array(y_pred)

def estimate_epochs(
    X, Y,
    model_func,
    model_params,
    patience=5,
    n_iter=50,
    batch_size=64,
    add_noise=False
):
    Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(X, Y, test_size=0.2, shuffle=False)
    if add_noise:
        Xtrain, Ytrain = augment_data(Xtrain, Ytrain)
    # scale the input data
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xvalid = scaler.transform(Xvalid)

    callback = EarlyStopping(monitor='val_loss', patience = patience, mode='min')
    epochs=300
    batch_size=batch_size

    n_epochs = []
    for _ in range(n_iter):
        model = model_func(**model_params)
        history = model.fit(
            Xtrain, Ytrain, epochs=epochs,
            validation_data = (Xvalid, Yvalid),
            callbacks=[callback],
            batch_size=batch_size,
            verbose=False
        )

        # use a trick to get the number of epochs the model has trained
        n_epochs_trained = len(history.history['loss'])
        # print(f"# epochs trained: {n_epochs_trained}")
        n_epochs.append(n_epochs_trained)
        print(f"{_}/{n_iter}", end='\r')
    
        
    print("mean number of epochs: {:.3}\nStd: {:.3f}".format(np.mean(n_epochs), np.std(n_epochs)))
    return np.mean(n_epochs)

def sample_station(df, threshold, seed=None):
    if seed is not None:
        np.random.seed(seed)
    df_n_data = df.groupby('skn').size().reset_index().rename(columns={0:"n_data"})
    sample_skn = df_n_data[df_n_data['n_data'] > threshold]['skn'].sample().values[0]
    df_station = df[df['skn'] == sample_skn].sort_values(['year', 'month'])
    print(f'Station with skn: {sample_skn} was chosen out of all stations with more than {threshold} historical (non-filled) rainfall observations.')
    print(f"There are {df_station.shape[0]} rainfall observations from this station.")
    return df_station