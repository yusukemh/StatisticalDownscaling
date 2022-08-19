# XGB
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2
import sys
sys.path.append('/home/yusukemh/github/yusukemh/StatisticalDownscaling/writeup')
from hyperparameters import XGB_PARAMS
from util import load_data, XGB
from config import C_COMMON, C_GRID, C_SINGLE, FILENAME

def main():
    
    columns = C_SINGLE
    for p in [i for i in range(50, 600, 50)]:
        ret_vals = []
        for item in XGB_PARAMS:
            skn = item['skn']
            print(item['skn'], item['params'])
            df_train, df_test = load_data(columns + C_COMMON, FILENAME)

            df_train = df_train[df_train['skn'] == skn]
            df_test = df_test[df_test['skn'] == skn]

            df_train = df_train.iloc[-p:]

            station_model = XGB(
                columns=columns,
                params=item['params'],
            )
            r = station_model.evaluate_by_station(df_train, df_test, skn=skn, n_iter=10)
            ret_vals.append(r)
        pd.DataFrame(ret_vals).to_csv(f'n_{p}_.csv')

if __name__ == '__main__':
    main()