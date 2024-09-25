import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class RidgeRegression:
    def __init__(self, lambda_param=1.0):
        self.lambda_param = lambda_param
        self.coefficients = None

    def fit(self, X, y):
        n_features = X.shape[1]
        identity = np.eye(n_features)
        # Ridge regression formula (eq 3.44)
        self.coefficients = np.linalg.inv(X.T @ X + self.lambda_param * identity) @ X.T @ y

    def predict(self, X):
        return X @ self.coefficients

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def select_lambda(X_train, y_train, X_val, y_val, lambda_values):
    best_lambda = None
    best_mse = float('inf')
    for lambda_param in lambda_values:
        model = RidgeRegression(lambda_param=lambda_param)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_val_pred)
        if mse < best_mse:
            best_mse = mse
            best_lambda = lambda_param
    return best_lambda

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
    df_lambda = np.array([np.trace(X @ np.linalg.inv(X.T @ X + lambda_param * np.eye(X.shape[1])) @ X.T) 
                          for lambda_param in lambda_values])

    plt.figure(figsize=(12, 8))
    for i in range(X.shape[1] - 1):  # Exclude intercept
        plt.plot(df_lambda, coeffs_normalized[:, i], label=feature_names[i], marker='o', markersize=3)
    
    plt.xlabel('df(λ)')
    plt.ylabel('Standardized Coefficients')
    plt.title('Ridge Regression Coefficients vs df(λ)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Add horizontal dotted line at y=0
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    
    # Add vertical dotted line at x=5
    plt.axvline(x=5, color='k', linestyle=':', alpha=0.5)
    
    # Set x-axis limits and ticks
    plt.xlim(0, 8)
    plt.xticks(range(0, 9, 2))
    
    plt.show()

if __name__ == '__main__':

    # Load and prepare data
    data = pd.read_csv("prostate_cancer.csv")
    y = data['lpsa']
    X = data.drop(['lpsa', 'train'], axis=1)
    feature_names = X.columns

    # Use the original train/test split
    train_mask = data['train'] == 'T'
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test_val = X[~train_mask]
    y_test_val = y[~train_mask]

    # Split the original test set into validation and new test sets
    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)

    # Standardize features
    for col in X.columns:
        if col != 'svi':  # Don't standardize binary predictor
            mean = X_train[col].mean()
            std = X_train[col].std()
            X_train[col] = (X_train[col] - mean) / std
            X_val[col] = (X_val[col] - mean) / std
            X_test[col] = (X_test[col] - mean) / std

    # Add constant term for intercept
    X_train = np.column_stack([np.ones(X_train.shape[0]), X_train])
    X_val = np.column_stack([np.ones(X_val.shape[0]), X_val])
    X_test = np.column_stack([np.ones(X_test.shape[0]), X_test])

    # Generate lambda values
    lambda_values = np.logspace(-2, 5, 100)

    # Select optimal lambda
    best_lambda = select_lambda(X_train, y_train, X_val, y_val, lambda_values)

    # Train final model with best lambda
    final_model = RidgeRegression(lambda_param=best_lambda)
    final_model.fit(X_train, y_train)

    # Evaluate on test set
    y_test_pred = final_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print(f"Best lambda: {best_lambda}")
    print(f"Test MSE: {test_mse}")

    # Plot ridge regression coefficients
    plot_ridge_coefficients(X_train, y_train, lambda_values, feature_names)
