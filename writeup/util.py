import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression as LR
from tqdm import tqdm

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
    
    def evaluate_by_station(self, df_train, df_test, skn):
        df_train_station = df_train[df_train['skn'] == skn]
        df_test_station = df_test[df_test['skn'] == skn]

        # convert to numpy
        x_train, x_test = np.array(df_train_station[self.columns]), np.array(df_test_station[self.columns])
        y_train, y_test = np.array(df_train_station['data_in']), np.array(df_test_station['data_in'])

        # scale the input and output
        x_train, x_test = self.transform_x(x_train, x_test)
        # y_train, y_test, y_scaler = self.transform_y(y_train, y_test)
        y_train, y_test = self.transform_y(y_train, y_test)

        # train the model with retrain_full = True
        history = self.train(x_train, y_train, verbose=0, retrain_full=True)

        # make prediction and scale
        y_pred = self.model.predict(x_test)
        y_pred = self.inverse_transform_y(y_pred)

        # scale y_test
        y_test = self.inverse_transform_y(y_test)
        
        return {
            "skn": skn,
            "rmse_nn": mean_squared_error(y_test, y_pred, squared=False),
            "mae_nn": mean_absolute_error(y_test, y_pred)
        }
        
    
    def evaluate(self, df_train, df_test):
        ret_vals = []
        for skn in tqdm(df_train['skn'].unique()):
            r = self.evaluate_by_station(df_train, df_test, skn)
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
        # scaler = MinMaxScaler(feature_range=(0,1))
        y_train = np.log(y_train + 1.)
        y_test = np.log(y_test + 1.)
        
        # NO MORE MinMax SCALING 
        # y_train = scaler.fit_transform(y_train.reshape(-1, 1))
        # y_test = scaler.transform(y_test.reshape(-1, 1))

        return y_train, y_test# , scaler
    
    def inverse_transform_y(self, y):
        # y = scaler.inverse_transform(y)
        y = np.power(np.e, y) - 1
        return y
    
    def train(self, x, y, verbose=0, retrain_full=False):
        # build the model
        self.model, batch_size = self.model_func(**self.params)
        
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
        history = self.model.fit(
            x, y,
            epochs=int(1e3),
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        if retrain_full:
            epochs = len(history.history['loss'])
            # rebuild the model
            self.model, batch_size = self.model_func(**self.params)
            callbacks = [EarlyStopping(monitor='loss', min_delta=0, patience=1e3, restore_best_weights=True)]
            history = self.model.fit(
                x, y,
                epochs=epochs,
                validation_split=0,
                callbacks=callbacks,
                batch_size=batch_size,
                verbose=0
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
        
    def train_and_save_base_model(self, model_path, params):
        """
        Train the base model and saves it
        """
        pass
            
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

    def train_base(self, x, y, retrain_full):
        pass
    
    def cross_val_predict_base(self, df, params, n_folds=5):
        """
        Run 5-fold cross validataion on entire dataset to choose hyperparameter for the base model
        """
        list_ytrue = []
        list_ypred = []
        for k in range(n_folds):
            df_train = df[df['inner_fold'] != k]
            df_test = df[df['inner_fold'] == k]
            
            x_train, x_test, y_train, y_test = self.split_scale(df_train, df_test)
            
            model, batch_size = self.model_func(**params)
            
            callbacks = [
                EarlyStopping(monitor='val_loss', min_delta=0, patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=10)
            ]

            history = model.fit(
                x_train, y_train,
                epochs=int(1e3),
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            y_pred = model.predict(x_test)
            list_ytrue.extend(np.power(np.e, y_test))
            list_ypred.extend(np.power(np.e, y_pred))
        # calculate the loss and return
        return {
            "rmse": mean_squared_error(list_ytrue, list_ypred, squared=False),
            "mae": mean_absolute_error(list_ytrue, list_ypred),
            # 'epochs': len(history.history['loss'])
        }
    
    def cross_val_predict_station(self, df, skn, n_folds=5):
        """
        Run 5-fold cross validation on a single station.
        """
        df_station = df[df['skn'] == skn]
        
        list_ytrue = []
        list_ypred = []
        for k in range(n_folds):
            df_train = df_station[df_station['inner_fold'] != k]
            df_test = df_station[df_station['inner_fold'] == k]
            
            x_train, x_test, y_train, y_test = self.split_scale(df_train, df_test)
                    
    def inverse_transform_y():
        pass
    
    def pre_train(x, y):
        pass