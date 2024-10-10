import numpy as np
import pandas as pd

from utils import logger


class L1RegularizedLogisticRegression:
    def __init__(self, learning_rate=0.01, lambda_param=1.0):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.theta = None
        self.u = 0  # Cumulative L1 penalty
        self.q = None  # Cumulative applied penalty for each feature

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def initialize(self, X):
        n_features = X.shape[1]
        self.theta = np.zeros(n_features + 1)  # +1 for the bias term
        self.q = np.zeros(n_features + 1)
        logger.info(f"Initialized theta and q with shape: {self.theta.shape}")

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.initialize(X)
        m, n = X.shape
        X_with_bias = np.column_stack([np.ones(m), X])

        logger.info(f"Starting fit with lambda: {self.lambda_param}")

        # Single pass through the data
        for i in range(m):
            x_i = X_with_bias[i]
            y_i = y[i]

            # Update cumulative L1 penalty
            self.u += self.learning_rate * self.lambda_param / m

            # Calculate gradient
            h = self.sigmoid(np.dot(self.theta, x_i))
            gradient = (h - y_i) * x_i

            # Update weights
            self.theta -= self.learning_rate * gradient

            # Apply L1 penalty
            for j in range(len(self.theta)):
                if self.theta[j] > 0:
                    self.theta[j] = max(0, self.theta[j] - (self.u + self.q[j]))
                elif self.theta[j] < 0:
                    self.theta[j] = min(0, self.theta[j] + (self.u - self.q[j]))

                # Update cumulative applied penalty
                self.q[j] += self.theta[j] - (
                    self.theta[j] + self.learning_rate * gradient[j]
                )

            if i % 1000 == 0:  # Log every 1000th iteration
                logger.debug(
                    f"Iteration {i}: Theta max: {np.max(self.theta)}, min: {np.min(self.theta)}"
                )
                logger.debug(
                    f"Iteration {i}: u: {self.u}, q max: {np.max(self.q)}, q min: {np.min(self.q)}"
                )

        logger.info(
            f"Finished fit. Final theta max: {np.max(self.theta)}, min: {np.min(self.theta)}"
        )
        logger.info(
            f"Final u: {self.u}, Final q max: {np.max(self.q)}, q min: {np.min(self.q)}"
        )

    def predict(self, X, threshold=0.5):
        if isinstance(X, pd.DataFrame):
            X = X.values

        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        prob = self.sigmoid(np.dot(X_with_bias, self.theta))
        predictions = (prob >= threshold).astype(int)
        logger.info(f"Made predictions. Positive cases: {np.sum(predictions)}")
        return predictions
