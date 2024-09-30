import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from lasso_regression import LassoRegression
from ridge_regression import RidgeRegression


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def load_and_split_data(filepath, target_col, test_size=0.1, val_size=0.1):
    data = pd.read_csv(filepath)
    y = data[target_col]
    X = data.drop([target_col], axis=1)

    # Check if 'train' column exists and drop it if present
    if "train" in X.columns:
        X = X.drop("train", axis=1)

    X = X.apply(pd.to_numeric, errors="coerce") # Convert to numeric, replacing any non-numeric values with NaN
    X = X.dropna(axis=1) # Drop any columns with NaN values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_data(X_train, X_test=None, X_val=None, binary_cols=None):
    binary_cols = binary_cols or []

    for col in X_train.columns:
        if col not in binary_cols:
            mean = X_train[col].mean()
            std = X_train[col].std()
            X_train[col] = (X_train[col] - mean) / std
            if X_test is not None:
                X_test[col] = (X_test[col] - mean) / std
            if X_val is not None:
                X_val[col] = (X_val[col] - mean) / std

    return X_train, X_test, X_val


def add_intercept(X):
    return np.column_stack([np.ones(X.shape[0]), X]) 


def display_statistics(column_names, coefficients, std_errors, z_scores):
    results = pd.DataFrame(
        {
            "Term": column_names,
            "Coefficient": coefficients,
            "Std. Error": std_errors,
            "Z Score": z_scores,
        }
    )
    print(results.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def plot_correlation_matrix(X_train):
    correlation_matrix = X_train.corr() 
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".3f")
    plt.title("Correlation Matrix (Training Set)")
    plt.show()


def select_lambda(X_train, y_train, X_val, y_val, lambda_values):
    best_lambda = None
    best_mse = float("inf")
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
        coeffs.append(model.coefficients[1:])

    coeffs = np.array(coeffs)
    std_devs = X[:, 1:].std(axis=0)
    coeffs_normalized = coeffs * std_devs

    df_lambda = np.array(
        [
            np.trace(
                X @ np.linalg.inv(X.T @ X + lambda_param * np.eye(X.shape[1])) @ X.T
            )  # 3.50 calculate df(λ)
            for lambda_param in lambda_values  # effective parameter over regularization
        ]
    )

    plt.figure(figsize=(12, 8))
    for i in range(X.shape[1] - 1):
        plt.plot(
            df_lambda,
            coeffs_normalized[:, i],
            label=feature_names[i + 1],
            marker="o",
            markersize=3,
        )

    plt.xlabel("df(λ)")
    plt.ylabel("Standardized Coefficients")
    plt.title("Ridge Regression Coefficients vs df(λ)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.axhline(y=0, color="k", linestyle=":", alpha=0.5)
    plt.axvline(x=5, color="k", linestyle=":", alpha=0.5)
    plt.xlim(0, 8)
    plt.xticks(range(0, 9, 2))
    plt.show()


def plot_lasso_coefficients(X, y, alphas, feature_names):
    coefs = []
    for alpha in alphas:
        lasso = LassoRegression()
        lasso.model.set_params(alphas=[alpha])
        lasso.fit(X, y)
        coefs.append(lasso.coefficients)

    coefs = np.array(coefs)

    plt.figure(figsize=(12, 8))

    s_values = np.sum(np.abs(coefs), axis=1) / np.sum(np.abs(coefs[0])) 

    for i, feature in enumerate(feature_names):
        plt.plot(s_values, coefs[:, i], label=feature)

    plt.xlabel("Shrinkage Factor s")
    plt.ylabel("Coefficients")
    plt.title("Lasso Coefficient Profiles")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.5)

    best_s = np.sum(np.abs(lasso.coefficients)) / np.sum(np.abs(coefs[0]))
    plt.axvline(x=best_s, color="r", linestyle="--", alpha=0.5, label="Chosen λ")

    plt.xlim(0, 1)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()
