import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

def cross_val_predict_for_nn(model, X, Y, callback, batch_size, epochs, verbose):
    kf = KFold(n_splits=5)
    y_pred = []

    for train_index, test_index in kf.split(X):
        Xtemp, Xtest = X[train_index], X[test_index]
        Ytemp, Ytest = Y[train_index], Y[test_index]
        
        Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(Xtemp, Ytemp, test_size=0.2, shuffle=True)
        
        # scale the input
        scaler = StandardScaler()
        Xtrain = scaler.fit_transform(Xtrain)
        Xvalid = scaler.transform(Xvalid)
        Xtest = scaler.transform(Xtest)
        
        if callback is None:
            model.fit(
                Xtrain, Ytrain, epochs=epochs,
                validation_data = (Xvalid, Yvalid),
                batch_size=batch_size,
                verbose=verbose
            )
            y_pred.extend(model.predict(Xtest).tolist())
        else:
            model.fit(
                Xtrain, Ytrain, epochs=epochs,
                validation_data = (Xvalid, Yvalid),
                callbacks=[callback],
                batch_size=batch_size,
                verbose=verbose
            )
            y_pred.extend(model.predict(Xtest).tolist())
    
    return np.array(y_pred)