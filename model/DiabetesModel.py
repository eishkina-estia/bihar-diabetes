import numpy as np


class DiabetesModel:
    def __init__(self, model):
        self.model = model

    def __preprocess(self, X):
        # your preprocessing logic: adding new featureds, etc.
        # example: X['weekday'] = X['pickup_datetime'].dt.weekday
        return X

    def __postprocess(self, raw_output):
        # your postprocessing logic: inverse transformation, etc.
        # example: np.expm1(raw_output)
        return np.round(raw_output)

    def fit(self, X, y):
        X_processed = self.__preprocess(X)
        self.model.fit(X_processed, y)
        return self

    def predict(self, X):
        X_processed = self.__preprocess(X)
        raw_output = self.model.predict(X_processed)
        return self.__postprocess(raw_output)
