"""
Module implementing capping
"""
import numpy as np
from sklearn.base import BaseEstimator
from typing import Tuple

class OutlierCapper():
    # Regular pipeline
    def __init__(self, factor: float=1.5):
        self.factor = factor
    
    def fit(self, X):
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower = q1 - self.factor * iqr
        self.upper = q3 + self.factor * iqr
    
    def transform(self, X) -> Tuple[np.array, np.array]:
        X_capped = np.where(X < self.lower, self.lower, X)
        X_capped = np.where(X_capped > self.upper, self.upper, X_capped)
        self.cap_label = ((X < self.lower) | (X > self.upper)).astype(int)
        return X_capped, self.cap_label

class Capper(BaseEstimator):
    # Deployment pipeline
    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X: np.array, y=None) -> BaseEstimator:
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower = q1 - self.factor * iqr
        self.upper = q3 + self.factor * iqr
        return self

    def transform(self, X: np.array, y=None) -> np.array:
        X_capped = np.where(X < self.lower, self.lower, X)
        X_capped = np.where(X_capped > self.upper, self.upper, X_capped)
        return X_capped


    
    


