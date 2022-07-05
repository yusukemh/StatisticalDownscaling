import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#tensorflow
import tensorflow as tf
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Model
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def define_model(
    input_dim=20, 
    lr=0.005, 
    activation='relu',
    n_units=1024,
    n_layers=2,
    dropout=0.5
):
    inputs = Input(shape=(input_dim,))
    x = Dense(units=n_units, activation=activation)(inputs)
    
    for i in range(n_layers - 1):
        if dropout:
            x = Dropout(rate=dropout)(x)
        x = Dense(units=n_units, activation=activation)(x)
    outputs = Dense(units=1, kernel_initializer=tf.keras.initializers.HeNormal, activation='softplus')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=[RootMeanSquaredError()]
    )
    
    return model


class Transfer_Model():
    def __init__(self, inputs: list, outputs: list):
        self.inputs = inputs
        self.outputs = outputs
        self.model = None
        self.model_params = None

        
    def set_model(self, model_params):
        self.model_params = model_params
        self.model = define_model(**model_params)
        
    def load_pretrained_model(self, model_path):
        return tf.keras.models.load_model(model_path)
        
    def save_pretrained_model(self, model_path):
        self.model.save(model_path)

    def train_test_split(self, df, fold, fine_tune:bool, skn=None):
        if fine_tune:
            assert skn is not None, "If fine tuning, make sure to pass skn"

        df_train, df_test = df[df['fold'] != fold], df[df['fold'] == fold]
        x_train, x_test = np.array(df_train[self.inputs]), np.array(df_test[self.inputs])
        y_train, y_test = np.array(df_train[self.outputs]), np.array(df_test[self.outputs])
        
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        if fine_tune:
            # split dataset
            df_train_station = df_train[df_train['skn'] == skn]
            df_test_station = df_test[df_test['skn'] == skn]
            
            # convert to numpy
            x_train_station = np.array(df_train_station[self.inputs])
            x_test_station = np.array(df_test_station[self.inputs])
            
            y_train_station = np.array(df_train_station[self.outputs])
            y_test_station = np.array(df_test_station[self.outputs])
            
            # scale
            x_train_station = scaler.transform(x_train_station)
            x_test_station = scaler.transform(x_test_station)
            
            return x_train, x_test, y_train, y_test, x_train_station, x_test_station, y_train_station, y_test_station
        else:
            return x_train, x_test, y_train, y_test
        
    def fine_tune(
        self, df, fold, skn, retrain_full,
        model_path,
        epochs=1000,
        batch_size=64,
        stopping_patience=20,
        lr=0.001,
        lr_factor=0.95,
        lr_patience=10,
        verbose=2,
        validation_split=0.2
    ):
        x_train, x_test, y_train, y_test, x_train_station, x_test_station, y_train_station, y_test_station = \
        self.train_test_split(df, fold, fine_tune=True, skn=skn)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', min_delta=0, patience=stopping_patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience)
        ]
        
        self.model = self.load_pretrained_model(model_path)
        self.model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=lr),
            loss='mse',
            metrics=[RootMeanSquaredError()]
        )
        history = self.model.fit(
            x_train_station, y_train_station,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=validation_split,
            callbacks=callbacks
        )
        
        if retrain_full:
            self.model = self.load_pretrained_model(model_path)
            epochs = len(history.history['loss'])
            callbacks = [
                EarlyStopping(monitor='loss', min_delta=0, patience=1e3, restore_best_weights=True)
            ]
            history = self.model.fit(
                x_train_station, y_train_station, 
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.0,
                callbacks=callbacks
            )
        return history
        
    def pre_train(
        self, df, fold:int, retrain_full,
        epochs=2000,
        batch_size=128,
        stopping_patience=20,
        lr=0.001,
        lr_factor=0.95,
        lr_patience=10,
        verbose=2,
        validation_split=0.2
    ):
        assert self.model is not None, "Please first provide the model structure with \"set_model\"."
        x_train, x_test, y_train, y_test = self.train_test_split(df, fold, fine_tune=False)
        callbacks = [
            EarlyStopping(monitor='val_loss', min_delta=0, patience=stopping_patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience)
        ]
        
        self.model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=lr),
            loss='mse',
            metrics=[RootMeanSquaredError()]
        )
        
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=validation_split,
            callbacks=callbacks
        )
        
        if retrain_full:
            self.model = define_model(**self.model_params)
            epochs = len(history.history['loss'])
            
            callbacks = [
                EarlyStopping(monitor='loss', min_delta=0, patience=1e3, restore_best_weights=True)
            ]
            
            history = self.model.fit(
                x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.0,
                callbacks=callbacks
            )
        return history
    