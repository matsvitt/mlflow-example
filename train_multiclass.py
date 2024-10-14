# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

from mlexample.multiclass_example import read_data, baseline_model, pre_process_data, generate_classes, create_data


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the sales data for the product
    _path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sales_data.csv")
    dfx=read_data(_path)    
    dfi1=pre_process_data(dfx)
    generate_classes(dfi1)
    X,y=create_data(dfi1)
    
    X_train, X_test, y_train, y_test, =train_test_split(X,y,test_size=0.2,shuffle=True)

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        model=baseline_model(X,y)
        model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        history=model.fit(X_train, y_train, epochs=40, batch_size=64, verbose=1, validation_data=(X_test,y_test))
        results = model.evaluate(X_test, y_test)
        print(f"Multiclass model {results}")

        mlflow.log_param("epochs", 40)
        mlflow.log_param("batch_size", 64)
        mlflow.log_metric("loss", results[0])
        mlflow.log_metric("accuracy", results[1])
        mlflow.sklearn.log_model(model, "model")
