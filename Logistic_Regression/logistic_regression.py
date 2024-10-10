import numpy as np
import pandas as pd

from utils import logger


class LogisticRegression:
    def __init__(self, learning_rate=0.01, lambda_param=0):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.theta = None

    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def initialize(self, X):

        n_features = X.shape[1]
        self.theta = np.zeros(n_features + 1)
        logger.info(f"Initialized theta with shape: {self.theta.shape}")

    def fit(self, X, y, regularized=False):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.initialize(X)
        m, n = X.shape
        X_with_bias = np.column_stack([np.ones(m), X])

        logger.info(
            f"Starting fit with lambda: {self.lambda_param}, regularized: {regularized}"
        )

        # Single pass through the data
        for i in range(m):
            x_i = X_with_bias[i]
            y_i = y[i]
            h = self.sigmoid(np.dot(self.theta, x_i))
            gradient = (h - y_i) * x_i
            if regularized:
                reg_term = (self.lambda_param / m) * self.theta[1:]
                gradient[1:] += reg_term
                if i % 100 == 0:  # Log every 100th iteration
                    logger.debug(
                        f"Iteration {i}: Regularization term max: {np.max(reg_term)}, min: {np.min(reg_term)}"
                    )
            self.theta -= self.learning_rate * gradient

            if i % 1000 == 0:  # Log every 1000th iteration
                logger.debug(
                    f"Iteration {i}: Theta max: {np.max(self.theta)}, min: {np.min(self.theta)}"
                )

        self.theta = np.clip(self.theta, -1e15, 1e15)
        logger.info(
            f"Finished fit. Final theta max: {np.max(self.theta)}, min: {np.min(self.theta)}"
        )

    def predict(self, X, threshold=0.5):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values

        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        prob = self.sigmoid(np.dot(X_with_bias, self.theta))
        predictions = (prob >= threshold).astype(int)
        logger.info(f"Made predictions. Positive cases: {np.sum(predictions)}")
        return predictions
