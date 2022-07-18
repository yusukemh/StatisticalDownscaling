import sherpa
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import sys
sys.path.append('/home/yusukemh/github/yusukemh/StatisticalDownscaling/writeup/')
from config import C_COMMON, C_GRID, C_SINGLE, FILENAME
from sklearn.model_selection import train_test_split
import time

def main():
    ######################
    df = pd.read_csv(FILENAME, usecols=C_COMMON + C_SINGLE).sort_values(['year', 'month'])
    columns = C_SINGLE
    column_type = 'single'
    ######################

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

    # define parameters for sherpa
    parameters = [
            sherpa.Choice('n_estimators', list(range(100, 310, 10))),
            sherpa.Continuous('learning_rate', [0.001, 0.1]),
            sherpa.Discrete('max_depth', [1, 10]),
    ]

    n_run = 500
    alg = sherpa.algorithms.RandomSearch(max_num_trials=n_run)
    study = sherpa.Study(parameters=parameters,
                         algorithm=alg,
                         lower_is_better=True)
    print('data processing complete')
    
    df_report = []
    for trial in study:
        start = time.time()
        line = '===============================================\n'
        params = {
            "n_estimators": trial.parameters['n_estimators'],
            "learning_rate": trial.parameters['learning_rate'],
            "max_depth": trial.parameters['max_depth'],
            "verbosity": 1
        }
        print(params)
        line += str(params) + '\n'

        dfs = []
        for k in range(5):
            df_train = df_inner_split.query(f'inner_fold != {k}')
            df_test = df_inner_split.query(f'inner_fold == {k}')

            x_train, x_test = np.array(df_train[columns]), np.array(df_test[columns])
            y_train, y_test = np.array(df_train['data_in']), np.array(df_test['data_in'])

            model = XGBRegressor(**params)
            model.fit(x_train, y_train)

            yhat = model.predict(x_test)
            rmse = mean_squared_error(y_test, yhat, squared=False)
            dfs.append(pd.DataFrame({'n_data': x_train.shape[0], 'rmse': rmse}, index=[k]))
        df_result = pd.concat(dfs)
        # calculate the weighted mean
        rmse = (df_result["n_data"] * df_result['rmse']).sum() / df_result['n_data'].sum()
        
        end = time.time()
        line += "RMSE on validation set: {:.6f}".format(rmse) + '\n'
        line += "elapsed time         : {:.3f}".format(end - start) + '\n'
        with open('progress.txt', 'a') as f:
            f.write(line)
        
        df_report.append(
            pd.DataFrame(params, index=[rmse])
        )
    pd.concat(df_report).to_csv(f'xgb_report_{n_run}_{column_type}.csv')
     
    
if __name__ == '__main__':
    main()