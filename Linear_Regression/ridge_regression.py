import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def plot_ridge_coefficients(X, y, lambda_values, feature_names):
    coeffs = []
    for lambda_param in lambda_values:
        model = RidgeRegression(lambda_param=lambda_param)
        model.fit(X, y)
        coeffs.append(model.coefficients[1:])  # Exclude intercept

    coeffs = np.array(coeffs)

    # Normalize coefficients
    std_devs = X[:, 1:].std(axis=0)
    coeffs_normalized = coeffs * std_devs

    # Calculate df(λ) - Effective degrees of freedom
    df_lambda = np.array(
        [
            np.trace(
                X @ np.linalg.inv(X.T @ X + lambda_param * np.eye(X.shape[1])) @ X.T
            )
            for lambda_param in lambda_values
        ]
    )

    plt.figure(figsize=(8, 8))
    for i in range(X.shape[1] - 1):  # Exclude intercept
        plt.plot(
            df_lambda,
            coeffs_normalized[:, i],
            label=feature_names[i],
            marker="o",
            markersize=3,
        )

    plt.xlabel("df(λ)")
    plt.ylabel("Standardized Coefficients")
    plt.title("Ridge Regression Coefficients vs df(λ)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Add horizontal dotted line at y=0
    plt.axhline(y=0, color="k", linestyle=":", alpha=0.5)

    # Add vertical dotted line at x=5
    plt.axvline(x=5, color="k", linestyle=":", alpha=0.5)

    # Set x-axis limits and ticks
    plt.xlim(0, 10)
    plt.xticks(range(0, 9, 2))

    plt.show()


if __name__ == "__main__":

    # Load and prepare data
    data = pd.read_csv("prostate_cancer.csv")
    y = data["lpsa"]
    X = data.drop(["lpsa", "train"], axis=1)
    feature_names = X.columns

    # Use the original train/test split
    train_mask = data["train"] == "T"
    X_train = X[train_mask]
    y_train = y[train_mask]

    # Standardize features
    for col in X_train.columns:
        if col != "svi":  # Don't standardize binary predictor
            mean = X_train[col].mean()
            std = X_train[col].std()
            X_train[col] = (X_train[col] - mean) / std

    # Add constant term for intercept
    X_train = np.column_stack([np.ones(X_train.shape[0]), X_train])

    # Generate lambda values
    lambda_values = np.logspace(3, -2, 100)[::-1]

    # Plot ridge regression coefficients
    plot_ridge_coefficients(X_train, y_train, lambda_values, feature_names)
