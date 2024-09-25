import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lasso_regression import LassoRegression
from linear_regression import LinearRegression
from ridge_regression import RidgeRegression
from utils import (add_intercept, display_statistics, load_and_split_data,
                   mean_squared_error, plot_correlation_matrix,
                   plot_lasso_coefficients, plot_ridge_coefficients,
                   select_lambda, standardize_data)

# Parameters
DATASET_PATH = "prostate_cancer.csv"
TARGET_COLUMN = "lpsa"
BINARY_COLUMNS = ["svi"]
TEST_SIZE = 0.1
VAL_SIZE = 0.1
LAMBDA_VALUES = np.logspace(-4, 5, 100)


def run_linear_regression(X_train, X_test, y_train, y_test):
    print("\n--- Linear Regression ---")
    model = LinearRegression()
    X_train_with_intercept = add_intercept(X_train)
    X_test_with_intercept = add_intercept(X_test)
    model.fit(X_train_with_intercept, y_train)

    y_train_pred = model.predict(X_train_with_intercept)
    y_test_pred = model.predict(X_test_with_intercept)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")

    std_errors, z_scores = model.calculate_statistics(
        X_train_with_intercept, y_train, y_train_pred
    )
    column_names = ["Intercept"] + list(X_train.columns)
    display_statistics(column_names, model.coefficients, std_errors, z_scores)

    plot_correlation_matrix(X_train)


def run_ridge_regression(X_train, X_val, X_test, y_train, y_val, y_test):
    print("\n--- Ridge Regression ---")
    X_train_with_intercept = add_intercept(X_train)
    X_val_with_intercept = add_intercept(X_val)
    X_test_with_intercept = add_intercept(X_test)

    best_lambda = select_lambda(
        X_train_with_intercept, y_train, X_val_with_intercept, y_val, LAMBDA_VALUES
    )
    print(f"Best lambda: {best_lambda}")

    final_model = RidgeRegression(lambda_param=best_lambda)
    final_model.fit(X_train_with_intercept, y_train)

    y_test_pred = final_model.predict(X_test_with_intercept)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f"Test MSE: {test_mse}")

    plot_ridge_coefficients(
        X_train_with_intercept,
        y_train,
        LAMBDA_VALUES,
        ["Intercept"] + list(X_train.columns),
    )


def run_lasso_regression(X_train, X_val, X_test, y_train, y_val, y_test):
    print("\n--- Lasso Regression ---")
    lasso = LassoRegression()
    lasso.fit(X_train, y_train)

    best_alpha = lasso.best_alpha
    print(f"Best alpha (lambda): {best_alpha}")

    y_test_pred = lasso.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f"Test MSE: {test_mse}")

    plot_lasso_coefficients(X_train, y_train, LAMBDA_VALUES, X_train.columns)


if __name__ == "__main__":
    # Load and prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
        DATASET_PATH, TARGET_COLUMN, TEST_SIZE, VAL_SIZE
    )

    # Standardize non-binary predictors
    X_train, X_test, X_val = standardize_data(
        X_train, X_test, X_val, binary_cols=BINARY_COLUMNS
    )

    # Run all regression methods
    run_linear_regression(X_train, X_test, y_train, y_test)
    run_ridge_regression(X_train, X_val, X_test, y_train, y_val, y_test)
    run_lasso_regression(X_train, X_val, X_test, y_train, y_val, y_test)
