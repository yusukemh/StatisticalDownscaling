import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2
import sys
sys.path.append('/home/yusukemh/github/yusukemh/StatisticalDownscaling/writeup')
from hyperparameters import NN_PARAMS
from util import load_data, NeuralNetwork
from config import C_COMMON, C_GRID, C_SINGLE, FILENAME

def define_model(
    input_dim=20,
    n_units=512,
    activation='selu',#selu
    learning_rate=0.00001,
    loss='mse',
    batch_size=64
):
    inputs = Input(shape=(input_dim))
    x = Dense(units=n_units, activation=activation, kernel_regularizer=L2(l2=0.01))(inputs)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=n_units, activation=activation, kernel_regularizer=L2(l2=0.01))(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=n_units, activation=activation, kernel_regularizer=L2(l2=0.01))(x)
    x = Dropout(rate=0.5)(x)# serves as regularization
    outputs = Dense(units=1, activation='softplus')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[RootMeanSquaredError()]
    )
    return model, batch_size

def main():
    
    columns = C_SINGLE
    for p in [i for i in range(50, 600, 50)]:
        ret_vals = []
        for item in NN_PARAMS:
            skn = item['skn']
            if p == 50: print(item['skn'], item['params'])
            df_train, df_test = load_data(columns + C_COMMON, FILENAME)

            df_train = df_train[df_train['skn'] == skn]
            df_test = df_test[df_test['skn'] == skn]

            # n_data = int(df_train.shape[0] * (p / 100.))
            # df_train = df_train.iloc[-n_data:]
            df_train = df_train.iloc[-p:]

            station_model = NeuralNetwork(
                columns=columns,
                params=item['params'],
                model_func=define_model
            )
            r = station_model.evaluate_by_station(df_train, df_test, skn=skn, n_iter=10, retrain_full=False)
            ret_vals.append(r)
        pd.DataFrame(ret_vals).to_csv(f'n_{p}_no_retrain.csv')

if __name__ == '__main__':
    main()