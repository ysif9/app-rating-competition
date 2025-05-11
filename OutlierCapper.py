import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.10, upper_quantile=0.90, factor=1.5):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.factor = factor

    def fit(self, X, y=None):
        # convert to 2D numpy array
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        lowers, uppers = [], []
        # for each column, compute Q1/Q3 and then low/high limits
        for col in arr.T:
            q1 = np.percentile(col, self.lower_quantile * 100)
            q3 = np.percentile(col, self.upper_quantile * 100)
            iqr = q3 - q1
            lowers.append(q1 - self.factor * iqr)
            uppers.append(q3 + self.factor * iqr)
        self.lowers_ = np.array(lowers)
        self.uppers_ = np.array(uppers)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        # clip each column to [low, up]
        return np.clip(arr, self.lowers_, self.uppers_)
