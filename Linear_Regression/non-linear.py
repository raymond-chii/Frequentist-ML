import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from linear_regression import LinearRegression
from utils import load_and_split_data, standardize_data, add_intercept

# Parameters
DATASET_PATH = "data/south_african_heart_disease.csv"
TARGET_COLUMN = "chd"
BINARY_COLUMNS = ["famhist"]
TEST_SIZE = 0.2  # Using 80% for training, 20% for testing

def add_nonlinear_terms(X):
    # Add squared terms
    for col in X.columns:
        if col not in BINARY_COLUMNS:
            X[f"{col}_squared"] = X[col] ** 2
    
    # Add inverse terms (avoiding division by zero)
    for col in X.columns:
        if col not in BINARY_COLUMNS:
            X[f"{col}_inverse"] = 1 / (X[col] + 1e-5)
    
    # Add interaction terms (only for non-binary columns)
    non_binary_cols = [col for col in X.columns if col not in BINARY_COLUMNS]
    for i in range(len(non_binary_cols)):
        for j in range(i+1, len(non_binary_cols)):
            col1, col2 = non_binary_cols[i], non_binary_cols[j]
            X[f"{col1}_{col2}_interaction"] = X[col1] * X[col2]
    
    return X

def main():
    # Load and prepare data
    X, _, _, y, _, _ = load_and_split_data(DATASET_PATH, TARGET_COLUMN, TEST_SIZE, 0)
    
    # Standardize non-binary predictors
    X, _, _ = standardize_data(X, None, None, binary_cols=BINARY_COLUMNS)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    
    # Run linear regression with original features
    model_original = LinearRegression()
    X_train_original = add_intercept(X_train)
    X_test_original = add_intercept(X_test)
    model_original.fit(X_train_original, y_train)
    y_pred_original = model_original.predict(X_test_original)
    mse_original = mean_squared_error(y_test, y_pred_original)
    r2_original = r2_score(y_test, y_pred_original)
    
    print("Results with original features:")
    print(f"MSE: {mse_original:.4f}, R2: {r2_original:.4f}")
    
    # Add nonlinear terms and interactions
    X_train_nonlinear = add_nonlinear_terms(X_train.copy())
    X_test_nonlinear = add_nonlinear_terms(X_test.copy())
    
    # Run linear regression with nonlinear features
    model_nonlinear = LinearRegression()
    X_train_nonlinear = add_intercept(X_train_nonlinear)
    X_test_nonlinear = add_intercept(X_test_nonlinear)
    model_nonlinear.fit(X_train_nonlinear, y_train)
    y_pred_nonlinear = model_nonlinear.predict(X_test_nonlinear)
    mse_nonlinear = mean_squared_error(y_test, y_pred_nonlinear)
    r2_nonlinear = r2_score(y_test, y_pred_nonlinear)
    
    print("\nResults with nonlinear and interaction terms:")
    print(f"MSE: {mse_nonlinear:.4f}, R2: {r2_nonlinear:.4f}")
    
    # Print the number of features in each model
    print(f"\nNumber of features in original model: {X_train_original.shape[1]}")
    print(f"Number of features in nonlinear model: {X_train_nonlinear.shape[1]}")

if __name__ == "__main__":
    main()