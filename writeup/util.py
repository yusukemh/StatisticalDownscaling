import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression as LR
from tqdm import tqdm
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import math

class Generator(Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, x_set, y_set, batch_size=256):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        # inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]  # Line A
        inds = self.indices.take(range(idx * self.batch_size, (idx + 1) * self.batch_size), mode='wrap')
        batch_x = self.x[inds]
        batch_y = self.y[inds]
        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


class LinearRegression():
    def __init__(self, columns):
        self.columns = columns
        
    def evaluate(self, df_train, df_test):
        ret_vals = []
        for skn in df_train['skn'].unique():
            df_train_station = df_train[df_train['skn'] == skn]
            df_test_station = df_test[df_test['skn'] == skn]
            
            x_train, x_test = np.array(df_train_station[self.columns]), np.array(df_test_station[self.columns])
            y_train, y_test = np.array(df_train_station['data_in']), np.array(df_test_station['data_in'])
            
            linear_regression = LR()
            linear_regression.fit(x_train, y_train)
            
            y_pred = linear_regression.predict(x_test)
            
            ret_vals.append(
                {
                    "skn": skn,
                    "rmse_lr": mean_squared_error(y_test, y_pred, squared=False),
                    "mae_lr": mean_absolute_error(y_test, y_pred)
                }
            )
        return pd.DataFrame(ret_vals)

class XGB():
    def __init__(self, params, columns):
        self.params = params
        self.columns = columns
        pass
    
    def cross_val_predict(self, df, skn, n_folds=5):
        assert 'inner_fold' in df.columns, 'define fold with column name "inner_fold"'
        df_station = df[df['skn'] == skn]
        
        list_ytrue = []
        list_ypred = []
        for k in range(n_folds):
            df_train = df_station[df_station['inner_fold'] != k]
            df_test = df_station[df_station['inner_fold'] == k]
            x_train, x_test = np.array(df_train[self.columns]), np.array(df_test[self.columns])
            y_train, y_test = np.array(df_train['data_in']), np.array(df_test['data_in'])
            
            model = XGBRegressor(**self.params)
            model.fit(x_train, y_train)
            
            y_pred = model.predict(x_test)
            
            list_ytrue.extend(y_test)
            list_ypred.extend(y_pred)
        return {
            "mae": mean_absolute_error(list_ytrue, list_ypred),
            "rmse": mean_squared_error(list_ytrue, list_ypred, squared=False)
        }
    
    def evaluate(self, df_train, df_test):
        ret_vals = []
        for skn in df_train['skn'].unique():
            df_train_station = df_train[df_train['skn'] == skn]
            df_test_station = df_test[df_test['skn'] == skn]

            x_train, x_test = np.array(df_train_station[self.columns]), np.array(df_test_station[self.columns])
            y_train, y_test = np.array(df_train_station['data_in']), np.array(df_test_station['data_in'])

            """
            # linear regression
            linear_regression = LinearRegression()
            linear_regression.fit(x_train, y_train)
            y_pred = linear_regression.predict(x_test)
            ret['mae_lr'] = mean_absolute_error(y_test, y_pred)
            ret['rmse_lr'] = mean_squared_error(y_test, y_pred, squared=False)
            """

            # xgb
            model = XGBRegressor(**self.params)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            ret_vals.append(
                {
                    "skn": skn,
                    "rmse_xgb": mean_squared_error(y_test, y_pred, squared=False),
                    "mae_xgb": mean_absolute_error(y_test, y_pred)
                }
            )
        return pd.DataFrame(ret_vals)

class NeuralNetwork():
    
    def __init__(self, model_func, params, columns):
        self.model_func = model_func
        self.params = params
        self.columns = columns
        pass
    
    def evaluate_by_station(self, df_train, df_test, skn, n_iter=1):
        rmse = []
        mae = []
        for iter in range(n_iter):
            df_train_station = df_train[df_train['skn'] == skn]
            df_test_station = df_test[df_test['skn'] == skn]

            # convert to numpy
            x_train, x_test = np.array(df_train_station[self.columns]), np.array(df_test_station[self.columns])
            y_train, y_test = np.array(df_train_station['data_in']), np.array(df_test_station['data_in'])

            # scale the input and output
            x_train, x_test = self.transform_x(x_train, x_test)
            y_train, y_test = self.transform_y(y_train, y_test)


            # train the model with retrain_full = True
            history = self.train(x_train, y_train, verbose=0, retrain_full=True)

            # make prediction and scale
            y_pred = self.model.predict(x_test)
            y_pred = self.inverse_transform_y(y_pred)

            # scale y_test
            y_test = self.inverse_transform_y(y_test)
            
            rmse.append(mean_squared_error(y_test, y_pred, squared=False))
            mae.append(mean_absolute_error(y_test, y_pred))
        
        return {
            "skn": skn,
            "n_iter": n_iter,
            "rmse_nn": np.mean(rmse),
            "mae_nn": np.mean(mae),
            "rmse_std_nn": np.std(rmse),
            "mae_std_nn": np.std(mae)
        }
        
    
    def evaluate(self, df_train, df_test, n_iter=1):
        ret_vals = []
        for skn in tqdm(df_train['skn'].unique()):
            r = self.evaluate_by_station(df_train, df_test, skn, n_iter)
            ret_vals.append(r)

        return pd.DataFrame(ret_vals)
            
            
    
    def cross_val_predict(self, df, skn, verbose=0, n_folds=5):
        assert 'inner_fold' in df.columns, 'define fold with column name "inner_fold"'
        df_station = df[df['skn'] == skn]
        
        list_ytrue = []
        list_ypred = []
        for k in range(n_folds):
            # split the dataset
            df_train = df_station[df_station['inner_fold'] != k]
            df_test = df_station[df_station['inner_fold'] == k]
            
            # convert to numpy
            x_train, x_test = np.array(df_train[self.columns]), np.array(df_test[self.columns])
            y_train, y_test = np.array(df_train['data_in']), np.array(df_test['data_in'])
            
            # scale the input and output
            x_train, x_test = self.transform_x(x_train, x_test)
            y_train, y_test = self.transform_y(y_train, y_test)
            
            # train the model
            history = self.train(x_train, y_train, verbose=0, retrain_full=False) # to speed up computation for hyperparaemter tuning
            
            # make prediction and scale
            y_pred = self.model.predict(x_test)
            y_pred = self.inverse_transform_y(y_pred)
            # scale y_test
            y_test = self.inverse_transform_y(y_test)
            
            # keep the record
            list_ytrue.extend(y_test)
            list_ypred.extend(y_pred)
        
        # calculate the loss and return
        return {
            "rmse": mean_squared_error(list_ytrue, list_ypred, squared=False),
            "mae": mean_absolute_error(list_ytrue, list_ypred),
            'epochs': len(history.history['loss'])
        }

    def transform_x(self, x_train, x_test):
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        return x_train, x_test
    
    def transform_y(self, y_train, y_test):
        y_train = np.log(y_train + 1.)
        y_test = np.log(y_test + 1.)

        return y_train, y_test# , scaler
    
    def inverse_transform_y(self, y):
        y = np.power(np.e, y) - 1
        return y
    
    def train(self, x, y, verbose=0, retrain_full=False):
        # split into train and validation
        # strictly speaking, this is not appropriate because scaler fit to the union of train/valid
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, shuffle=False)
        # build the model
        self.model, batch_size = self.model_func(**self.params)
        # set up callbacks
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
        
        # set up the generators
        train_datagen = Generator(x_train, y_train, batch_size)
        valid_datagen = Generator(x_valid, y_valid, batch_size)
        
        history = self.model.fit(
            x=train_datagen,
            # y=None: x is tf.keras.Sequence so no need to specify
            steps_per_epoch=np.ceil(len(x_train)/batch_size),
            validation_data=valid_datagen,
            validation_steps=np.ceil(len(x_valid)/batch_size),
            epochs=int(1e3),
            callbacks=callbacks,
            verbose=0
        )
        
        if retrain_full:
            epochs = len(history.history['loss'])
            train_datagen = Generator(x, y, batch_size)
            # rebuild the model
            self.model, batch_size = self.model_func(**self.params)
            callbacks = [EarlyStopping(monitor='loss', min_delta=0, patience=1e3, restore_best_weights=True)]
            history = self.model.fit(
                x=train_datagen,
                # y=None: x is tf.keras.utils.Sequence so no need to specify
                steps_per_epoch=np.ceil(len(x) / batch_size),
                epochs=epochs,
                callbacks=callbacks,
                verbose=0,
            )
        return history        


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

def load_data(usecols, filename):
    """
    Loads dataset and splits into train and test.
    It also splits training dataset into 5 folds as a column named 'inner_fold'
    """
    df = pd.read_csv(filename, usecols=usecols).sort_values(['year', 'month'])
    df_train = df.query('fold != 4')
    df_test = df.query('fold == 4')
    assert (sorted(df_train['skn'].unique()) == sorted(df_test['skn'].unique()))
    
    df_train = assign_inner_fold(df_train)
    
    return df_train, df_test

class TransferModel():
    def __init__(self, model_func, columns):
        self.model_func = model_func
        self.columns = columns
            
    def split_scale(self, df_train, df_test):
        # convert to numpy
        x_train, x_test = np.array(df_train[self.columns]), np.array(df_test[self.columns])
        y_train, y_test = np.array(df_train['data_in']), np.array(df_test['data_in'])
        
        # scale x
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        # scale y
        y_train = np.log(y_train + 1.)
        y_test = np.log(y_test + 1.)
        
        return x_train, x_test, y_train, y_test
    
    def train_base(self, x, y, params, retrain_full=False, verbose=0):
        model, batch_size = self.model_func(**params)
        
        callbacks = [
                EarlyStopping(monitor='val_loss', min_delta=0, patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=10)
            ]

        history = model.fit(
            x, y,
            epochs=int(1e3),
            validation_split=0.2,
            callbacks=callbacks,
            batch_size=batch_size,
            verbose=verbose
        )
        
        if retrain_full:
            # retraining the base model is harmful: the model overfits
            epochs = len(history.history['loss'])
            
            model, batch_size = self.model_func(**params)
            callbacks = [
                EarlyStopping(monitor='loss', min_delta=0, patience=1e3, restore_best_weights=True),
                LearningRateScheduler(lambda epoch: history.history['lr'][epoch], verbose=0)
            ]
            
            history = model.fit(
                x, y,
                epochs=epochs,
                validation_split=0,
                callbacks=callbacks,
                batch_size=batch_size,
                verbose=verbose
            )
        return model, history   
    
    def cross_val_predict_base(self, df, params, n_folds=5):
        """
        Run 5-fold cross validataion on entire dataset to choose hyperparameter for the base model
        """
        list_ytrue = []
        list_ypred = []
        for k in range(n_folds):
            print(f"fold {k}")
            df_train = df[df['inner_fold'] != k]
            df_test = df[df['inner_fold'] == k]
            
            x_train, x_test, y_train, y_test = self.split_scale(df_train, df_test)
            
            model, history = self.train_base(x_train, y_train, params, retrain_full=False, verbose=0)# do not retrain full to save time. also, it hurts
            
            y_pred = model.predict(x_test)
            list_ytrue.extend(np.power(np.e, y_test))
            list_ypred.extend(np.power(np.e, y_pred))
        # calculate the loss and return
        return {
            "rmse": mean_squared_error(list_ytrue, list_ypred, squared=False),
            "mae": mean_absolute_error(list_ytrue, list_ypred),
        }