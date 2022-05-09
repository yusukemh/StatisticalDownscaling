import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from tensorflow.keras.callbacks import EarlyStopping

# from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed

def augment_data(X, Y):
    # make sure to apply this only on the training dataset
    # add noise to X
    new_X = [X]
    new_Y = [Y]
    for _ in range(5):
        noise = np.random.random(X.shape) * 0.001
        new_X.append(X + noise)
        new_Y.append(Y)
        
    
    return (np.vstack(new_X), np.array(new_Y).flatten())

def cross_val_predict_for_nn(
    model, X, Y,
    callback, batch_size, epochs,
    early_stopping=True,
    val_size=0.2,
    add_noise=False,
    verbose=False
):
    kf = KFold(n_splits=5)
    y_pred = []

    for train_index, test_index in kf.split(X):
        if early_stopping:
            Xtemp, Xtest = X[train_index], X[test_index]
            Ytemp, Ytest = Y[train_index], Y[test_index]

            Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(Xtemp, Ytemp, test_size=0.2, shuffle=True)

            # scale the input
            scaler = StandardScaler()
            Xtrain = scaler.fit_transform(Xtrain)
            Xvalid = scaler.transform(Xvalid)
            Xtest = scaler.transform(Xtest)
            # if early_stopping is true, then callback must not be None
            model.fit(
                Xtrain, Ytrain, epochs=epochs,
                validation_data = (Xvalid, Yvalid),
                callbacks=[callback],
                batch_size=batch_size,
                verbose=verbose
            )
            y_pred.extend(model.predict(Xtest).tolist())
        else: #if no early stopping
            Xtrain, Xtest = X[train_index], X[test_index]
            Ytrain, Ytest = Y[train_index], Y[test_index]
            if add_noise:
                Xtrain, Ytrain = augment_data(Xtrain, Ytrain)
            
            scaler = StandardScaler()
            Xtrain = scaler.fit_transform(Xtrain)
            Xtest = scaler.transform(Xtest)
            if callback is None:
                model.fit(
                    Xtrain, Ytrain, epochs=epochs,
                    batch_size=batch_size,
                    verbose=verbose
                )
            else:
                model.fit(
                    Xtrain, Ytrain, epochs=epochs,
                    callbacks=[callback],
                    batch_size=batch_size,
                    verbose=verbose
                )
            y_pred.extend(model.predict(Xtest).tolist())
            
            
    
    return np.array(y_pred)

def estimate_epochs(
    X, Y,
    model,
    patience=5,
    n_iter=50
):
    #model = define_model(input_dim=len(columns), lr=0.005)
    Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(X, Y, test_size=0.2, shuffle=False)
    # scale the input data
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xvalid = scaler.transform(Xvalid)

    # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = patience, mode='min')
    callback = EarlyStopping(monitor='val_loss', patience = patience, mode='min')
    epochs=300
    batch_size=64

    n_epochs = []
    for _ in range(n_iter):
        history = model.fit(
            Xtrain, Ytrain, epochs=epochs,
            validation_data = (Xvalid, Yvalid),
            callbacks=[callback],
            batch_size=batch_size,
            verbose=False
        )

        # use a trick to get the number of epochs the model has trained
        n_epochs_trained = len(history.history['loss'])
        # print(f"# epochs trained: {n_epochs_trained}")
        n_epochs.append(n_epochs_trained)
        print(f"{_}/{n_iter}", end='\r')
    
        
    print("mean number of epochs: {:.3}\nStd: {:.3f}".format(np.mean(n_epochs), np.std(n_epochs)))
    return np.mean(n_epochs)

def sample_station(df, threshold, seed=None):
    if seed is not None:
        np.random.seed(seed)
    df_n_data = df.groupby('skn').size().reset_index().rename(columns={0:"n_data"})
    sample_skn = df_n_data[df_n_data['n_data'] > threshold]['skn'].sample().values[0]
    df_station = df[df['skn'] == sample_skn].sort_values(['year', 'month'])
    print(f'Station with skn: {sample_skn} was chosen out of all stations with more than {threshold} historical (non-filled) rainfall observations.')
    print(f"There are {df_station.shape[0]} rainfall observations from this station.")
    return df_station