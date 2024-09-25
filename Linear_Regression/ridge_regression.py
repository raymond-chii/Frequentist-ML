# ridge_regression.py
import numpy as np


class RidgeRegression:
    def __init__(self, lambda_param=1.0):
        self.lambda_param = lambda_param
        self.coefficients = None

    def fit(self, X, y):
        n_features = X.shape[1]
        identity = np.eye(n_features)
        # Ridge regression formula (eq 3.44)
        self.coefficients = (
            np.linalg.inv(X.T @ X + self.lambda_param * identity) @ X.T @ y
        )

    def predict(self, X):
        return X @ self.coefficients
