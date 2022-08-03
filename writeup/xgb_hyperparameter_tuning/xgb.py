import sherpa
import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('/home/yusukemh/github/yusukemh/StatisticalDownscaling/writeup')
from config import C_COMMON, C_GRID, C_SINGLE, FILENAME
from util import load_data

from sklearn.linear_model import LinearRegression
import time

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

def main():
    columns, col_type = C_SINGLE, 'single'
    n_run = 500
    df_train, _ = load_data(columns + C_COMMON, FILENAME)
    
    # set up sherpa
    #====================================================================
    parameters = [
        sherpa.Choice('n_estimators', list(range(100, 310, 10))),
        sherpa.Continuous('learning_rate', [0.001, 0.1]),
        sherpa.Discrete('max_depth', [1, 10]),
    ]
    study = sherpa.Study(
        parameters=parameters,
        algorithm=sherpa.algorithms.RandomSearch(max_num_trials=n_run),
        lower_is_better=True
    )
    #====================================================================

    dfs = []
    for i, trial in enumerate(study):
        start = time.time()
        #obtain hyperparameters
        params = {key: val for key, val in trial.parameters.items()}
        params['input_dim'] = len(columns)
        line = '===============================================\n'
        line += str(params) + '\n'
        
        for skn in tqdm(df_train['skn'].unique()):
            model = XGB(params=params, columns=columns)
            ret = model.cross_val_predict(df_train, skn)
            df = pd.DataFrame()
            df['trial_id'] = i,
            df['rmse'] = [ret['rmse']]
            df['mae'] = [ret['mae']]
            df['skn'] = [skn]
            for key, val in params.items():
                df[key] = [val]
            dfs.append(df)        
        end = time.time()
        pd.concat(dfs).to_csv(f'xgb_report_{n_run}_{col_type}.csv')
        end = time.time()
        line += "elapsed time         : {:.3f}".format(end - start) + '\n'
        with open('progress.txt', 'a') as f:
            f.write(line)

if __name__ == '__main__':
    main()