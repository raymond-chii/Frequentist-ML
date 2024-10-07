import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, lambda_param=0):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.theta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize(self, X):
        n_features = X.shape[1]
        self.theta = np.zeros(n_features + 1)  # +1 for the bias term

    def calculate_gradient(self, x, y):
        h = self.sigmoid(np.dot(self.theta, x))
        gradient = (h - y) * x  # Note: (h - y) for descent, not (y - h)
        return gradient
    
    def calculate_gradient_l2(self, X, y):
        m = X.shape[0]
        h = self.sigmoid(np.dot(X, self.theta))
        gradient = np.dot(X.T, (h - y)) / m
        gradient[1:] += (self.lambda_param / m) * self.theta[1:]
        return gradient

    def fit(self, X, y):
        self.initialize(X)
        m, n = X.shape
        X_with_bias = np.column_stack([np.ones(m), X])

        for _ in range(n):
            for i in range(m):
                x_i = X_with_bias[i]
                y_i = y[i]
                gradient = self.calculate_gradient(x_i, y_i)
                self.theta -= self.learning_rate * gradient  # Note: -= for descent

    def fit_l2(self, X, y):
        self.initialize(X)
        m, n = X.shape
        X_with_bias = np.column_stack([np.ones(m), X])

        for _ in range(n):
            for i in range(m):
                x_i = X_with_bias[i]
                y_i = y[i]
                gradient = self.calculate_gradient_l2(x_i, y_i)
                self.theta -= self.learning_rate * gradient

    def predict(self, X, threshold=0.5):
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        prob = self.sigmoid(np.dot(X_with_bias, self.theta))

        return (prob >= threshold).astype(int)
