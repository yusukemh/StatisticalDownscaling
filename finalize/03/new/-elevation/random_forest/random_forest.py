# basic libraries
import numpy as np
import pandas as pd

# sklearn
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

#others
import time
import sherpa

BASE_DIR='/home/yusukemh/github/yusukemh/StatisticalDownscaling/dataset'

LABELS=[
    'air2m', 'air1000_500', 'hgt500', 'hgt1000', 'omega500',
    'pottemp1000-500', 'pottemp1000-850', 'pr_wtr', 'shum-uwnd-700',
    'shum-uwnd-925', 'shum-vwnd-700', 'shum-vwnd-950', 'shum700', 'shum925', 'skt', 'slp',
    'lat', 'lon', 'season_wet'
]

def main():
    # load datasets
    
    df = pd.read_csv('/home/yusukemh/sadow_lts/personal/yusukemh/pi_casc/processed_datasets/dataset_6grid.csv')
    # split
    df_train = df.query('year < 1984')
    df_valid = df.query('1984 <= year < 1997')
    df_test = df.query('1997 <= year')

    Xtrain = np.array(df_train[LABELS])
    Ytrain = np.array(df_train["data_in"])

    Xvalid = np.array(df_valid[LABELS])
    Yvalid = np.array(df_valid["data_in"])

    Xtest = np.array(df_test[LABELS])
    Ytest = np.array(df_test["data_in"])
    
    file_name = './RFR_elevation.txt'

    parameters = [
        sherpa.Choice('n_estimators', list(range(100, 310, 10))),
        sherpa.Discrete('min_samples_split', [2, 6])
    ]
    alg = sherpa.algorithms.RandomSearch(max_num_trials=50)
    study = sherpa.Study(parameters=parameters,
                         algorithm=alg,
                         lower_is_better=True)

    for trial in study:
        start = time.time()
        line = '===============================================\n'
        params = {
            "n_estimators": trial.parameters['n_estimators'],
            "max_depth": None,
            "min_samples_split": trial.parameters["min_samples_split"],
            "max_samples": 0.9,
            "verbose": True,
            "n_jobs": -1,
        }
        print(params)
        line += str(params) + '\n'
        model = RandomForestRegressor(**params)
        model.fit(Xtrain, Ytrain)
        training_error = mean_squared_error(Ytrain, model.predict(Xtrain))
        validation_error = mean_squared_error(Yvalid, model.predict(Xvalid))
        study.add_observation(
            trial=trial,
            iteration=1,
            objective=validation_error,
            context={'training_error': training_error}
        )
        end = time.time()
        line += "MSE on training set  : {:.6f}".format(training_error) + '\n'
        line += "MSE on validation set: {:.6f}".format(validation_error) + '\n'
        line += "elapsed time         : {:.3f}".format(end - start) + '\n'

        with open(file_name, 'a') as f:
            f.write(line)

        study.finalize(trial)

    print(study.get_best_result())

if __name__ == "__main__":
    main()