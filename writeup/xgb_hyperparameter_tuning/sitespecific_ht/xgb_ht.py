import sherpa
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import time

import sys
sys.path.append('/home/yusukemh/github/yusukemh/StatisticalDownscaling/writeup')
from config import C_COMMON, C_GRID, C_SINGLE, FILENAME

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv(FILENAME, usecols=C_COMMON + C_SINGLE).sort_values(['year', 'month'])
columns = C_SINGLE
column_type = 'single'

# we use the last 1/5 data as the heldout clean dataset. We do not use this fold for any use except for just reporting the result.
df_train_outer = df.query('fold != 4')
df_test_outer = df.query('fold == 4')
assert (sorted(df_test_outer['skn'].unique()) == sorted(df_train_outer['skn'].unique()))
# print(f"{df_train_outer.shape}, {df_test_outer.shape}")

# split the trainig data into 5 folds for inner cross validation
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

df_inner_split = assign_inner_fold(df_train_outer)

# randomly choose params
parameters = [
    sherpa.Choice('n_estimators', list(range(100, 310, 10))),
    sherpa.Continuous('learning_rate', [0.001, 0.1]),
    sherpa.Discrete('max_depth', [1, 10]),
]

n_run = 500
alg = sherpa.algorithms.RandomSearch(max_num_trials=n_run)
study = sherpa.Study(
    parameters=parameters,
    algorithm=alg,
    lower_is_better=True
)

dfs = []
for i_trial, trial in enumerate(study):
    start = time.time()
    line = '===============================================\n'
    params = {
        "n_estimators": trial.parameters['n_estimators'],
        "learning_rate": trial.parameters['learning_rate'],
        "max_depth": trial.parameters['max_depth'],
        "verbosity": 1
        
    }
    for skn in df_inner_split['skn'].unique():
        df_station = df_inner_split[df_inner_split['skn'] == skn]
        
        # residual sum of squares
        rss = []
        for inner_fold in range(5):
            df_train_station = df_station[df_station['inner_fold'] != inner_fold]
            df_test_station = df_station[df_station['inner_fold'] == inner_fold]

            x_train, y_train = np.array(df_train_station[columns]), np.array(df_train_station['data_in'])
            x_test, y_test = np.array(df_test_station[columns]), np.array(df_test_station['data_in'])

            model = XGBRegressor(**params)
            model.fit(x_train, y_train)
            yhat = model.predict(x_test)
            mse = mean_squared_error(y_test, yhat)
            
            rss.append(mse * df_train_station.shape[0])
        rmse = np.sqrt(np.array(rss).sum() / df_station.shape[0])
        # rmse = np.sqrt(np.array(rss).mean())
        df = pd.DataFrame(
            params, index=[rmse]
        )
        df['skn'] = [skn]
        
        dfs.append(df)
    if i_trial % 50 == 0:
        pd.concat(dfs).to_csv(f'xgb_sitespecific_{n_run}_{column_type}.csv')
    end = time.time()
    line += "RMSE on validation set: {:.6f}".format(rmse) + '\n'
    line += "elapsed time         : {:.3f}".format(end - start) + '\n'
    with open('progress.txt', 'a') as f:
        f.write(line)
pd.concat(dfs).to_csv(f'xgb_sitespecific_{n_run}_{column_type}.csv')

            
            
        
        