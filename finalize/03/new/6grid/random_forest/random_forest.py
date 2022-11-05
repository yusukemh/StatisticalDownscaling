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

LABELS = [
    'air2m_0', 'air1000_500_0', 'hgt500_0', 'hgt1000_0', 'omega500_0', 'pottemp1000-500_0', 'pottemp1000-850_0', 'pr_wtr_0', 'shum-uwnd-700_0', 'shum-uwnd-925_0', 'shum-vwnd-700_0', 'shum-vwnd-950_0', 'shum700_0', 'shum925_0', 'skt_0', 'slp_0', 'air2m_1', 'air1000_500_1', 'hgt500_1', 'hgt1000_1', 'omega500_1', 'pottemp1000-500_1', 'pottemp1000-850_1', 'pr_wtr_1', 'shum-uwnd-700_1', 'shum-uwnd-925_1', 'shum-vwnd-700_1', 'shum-vwnd-950_1', 'shum700_1', 'shum925_1', 'skt_1', 'slp_1', 'air2m_2', 'air1000_500_2', 'hgt500_2', 'hgt1000_2', 'omega500_2', 'pottemp1000-500_2', 'pottemp1000-850_2', 'pr_wtr_2', 'shum-uwnd-700_2', 'shum-uwnd-925_2', 'shum-vwnd-700_2', 'shum-vwnd-950_2', 'shum700_2', 'shum925_2', 'skt_2', 'slp_2', 'air2m_3', 'air1000_500_3', 'hgt500_3', 'hgt1000_3', 'omega500_3', 'pottemp1000-500_3', 'pottemp1000-850_3', 'pr_wtr_3', 'shum-uwnd-700_3', 'shum-uwnd-925_3', 'shum-vwnd-700_3', 'shum-vwnd-950_3', 'shum700_3', 'shum925_3', 'skt_3', 'slp_3', 'air2m_4', 'air1000_500_4', 'hgt500_4', 'hgt1000_4', 'omega500_4', 'pottemp1000-500_4', 'pottemp1000-850_4', 'pr_wtr_4', 'shum-uwnd-700_4', 'shum-uwnd-925_4', 'shum-vwnd-700_4', 'shum-vwnd-950_4', 'shum700_4', 'shum925_4', 'skt_4', 'slp_4', 'air2m_5', 'air1000_500_5', 'hgt500_5', 'hgt1000_5', 'omega500_5', 'pottemp1000-500_5', 'pottemp1000-850_5', 'pr_wtr_5', 'shum-uwnd-700_5', 'shum-uwnd-925_5', 'shum-vwnd-700_5', 'shum-vwnd-950_5', 'shum700_5', 'shum925_5', 'skt_5', 'slp_5', 'lat', 'lon', 'elevation', 'season_wet'
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