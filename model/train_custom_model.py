import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import common
import os, pickle

from model.DiabetesModel import DiabetesModel

DB_PATH = common.CONFIG['paths']['db_path']
MODEL_CUSTOM_PATH = common.CONFIG['paths']['model_custom_path']

def load_train_data(path):
    print(f"Reading train data from the database: {path}")
    con = sqlite3.connect(path)
    data_train = pd.read_sql('SELECT * FROM train', con)
    con.close()
    X = data_train.drop(columns=['target'])
    y = data_train['target']
    return X, y


def train_model(X, y):
    print(f"Building a model")
    model = DiabetesModel(LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)
    score = mean_squared_error(y, y_pred)
    print(f"Score on train data {score:.2f}")
    return model

def persist_model(model, path):
    print(f"Persisting the model to {path}")
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(path, "wb") as file:
        pickle.dump(model, file)
    print(f"Done")

if __name__ == "__main__":

    X_train, y_train = load_train_data(DB_PATH)
    model = train_model(X_train, y_train)
    persist_model(model, MODEL_CUSTOM_PATH)