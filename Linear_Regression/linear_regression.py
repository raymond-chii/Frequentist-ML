import numpy as np


class LinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        # Fit the linear regression model using the normal equation.
        self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        # Predict using the linear regression model.
        return X @ self.coefficients

    def calculate_mse(self, y_true, y_pred):
        # Calculate Mean Squared Error.
        mse = np.mean((y_true - y_pred) ** 2)
        return mse

    def calculate_statistics(self, X_train, y_train, y_train_hat):
        # Calculate residuals, variance, standard errors, and Z-scores.
        residuals = y_train - y_train_hat
        n, p = X_train.shape
        sigma_squared = np.sum(residuals**2) / (n - p)
        var_coeffs = sigma_squared * np.linalg.inv(X_train.T @ X_train)
        std_errors = np.sqrt(np.diag(var_coeffs))
        z_scores = self.coefficients / std_errors
        return std_errors, z_scores
