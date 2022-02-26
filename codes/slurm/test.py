# basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans

#others
from xgboost import XGBRegressor
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import time
import xarray as xr
import sherpa

def main():
    # Variables from config file
    BASE_DIR = "/home/yusukemh/github/yusukemh/StatisticalDownscaling/dataset"

    # Load the dataset
    df_metadata = pd.read_excel(f"{BASE_DIR}/FilledDataset2012.xlsx", sheet_name="Header")
    df_data_original = pd.read_csv(f"{BASE_DIR}/dataset.csv")

    # make sure there is no NaN value
    assert df_data_original.isnull().values.any() == False
    print(f"There are {df_data_original.shape[0]} samples.")
    print(
        "Each sample is associated with lat and lon coordinates.\n" + 
        "Use only the closest observation to represent each field, from 16 different NetCDF files.", )

    df_combined = df_data_original.merge(right=df_metadata[["SKN", "ElevFT"]], left_on="skn", right_on="SKN")
    df_clean = (
        df_combined.drop(
            labels=["lat", "lon", "year", "month", "SKN", "skn", "Lon_DD_updated"],
            axis=1
        ).rename(
            columns={"Lat_DD": "lat", "Lon_DD": "lon", "ElevFT": "elev"}
        )
    )

    # split the dataset without "elev"
    X = np.array(df_clean.drop(labels=["data_in", "elev"], axis=1))
    Y = np.array(df_clean["data_in"])

    Xtemp, Xtest, Ytemp, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
    Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(Xtemp, Ytemp, test_size=0.25, random_state=42)

    parameters = [
        sherpa.Choice('n_estimators', list(range(50, 310, 10))),
        sherpa.Discrete('min_samples_split', [2, 10])
    ]
    alg = sherpa.algorithms.RandomSearch(max_num_trials=1)
    study = sherpa.Study(parameters=parameters,
                         algorithm=alg,
                         lower_is_better=True)

    for trial in study:
        params = {
            "n_estimators": trial.parameters['n_estimators'],
            "max_depth": None,
            "min_samples_split": trial.parameters["min_samples_split"],
            "verbose": True,
            "n_jobs": -1
        }
        print(params)
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

        study.finalize(trial)

    print(study.get_best_result())

if __name__ == "__main__":
    main()