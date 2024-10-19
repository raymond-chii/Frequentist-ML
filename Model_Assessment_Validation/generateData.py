import numpy as np


def generateData(n_samples=50, n_predictors=5000):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_predictors)
    y = np.random.choice([-1, 1], size=n_samples)
    return X, y
