import numpy as np
import pandas as pd
from linear_regression import LinearRegression
from utils import load_and_split_data, standardize_data, add_intercept, display_statistics, plot_correlation_matrix

# Parameters: Change these to run with different datasets or configurations
DATASET_PATH = "prostate_cancer.csv"  # Path to your dataset
TARGET_COLUMN = "lpsa"                # Target column for prediction
BINARY_COLUMNS = ["svi"]              # Columns that shouldn't be standardized
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Load and prepare data
X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(DATASET_PATH, TARGET_COLUMN, TEST_SIZE, VAL_SIZE)

# **Drop any non-numeric columns (like 'train')** before standardization
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])
X_val = X_val.select_dtypes(include=[np.number])

# Standardize non-binary predictors
X_train, X_test, X_val = standardize_data(X_train, X_test, X_val, binary_cols=BINARY_COLUMNS)

# Add intercept column
X_train_with_intercept = add_intercept(X_train)
X_test_with_intercept = add_intercept(X_test)

# Train linear regression model
model = LinearRegression()
model.fit(X_train_with_intercept, y_train)

# Predictions
y_train_hat = model.predict(X_train_with_intercept)
y_test_hat = model.predict(X_test_with_intercept)

# Calculate MSE
mse_train = model.calculate_mse(y_train, y_train_hat)
mse_test = model.calculate_mse(y_test, y_test_hat)
print(f"Train MSE: {mse_train:.4f}")
print(f"Test MSE: {mse_test:.4f}")

# Calculate residuals and statistics
std_errors, z_scores = model.calculate_statistics(X_train_with_intercept, y_train, y_train_hat)

# Display statistics (Table 3.2)
column_names = ["Intercept"] + list(X_train.columns)
display_statistics(column_names, model.coefficients, std_errors, z_scores)

# Plot correlation matrix (Table 3.1)
plot_correlation_matrix(X_train)
