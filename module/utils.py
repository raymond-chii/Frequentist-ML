import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_split_data(filepath, target_col, test_size=0.2, val_size=0.1):
    """Load data from a CSV file and split into train, validation, and test sets."""
    data = pd.read_csv(filepath)
    y = data[target_col]
    X = data.drop([target_col], axis=1)
    
    # Split the data into train (80%) and temp (20% for validation and test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=42)
    
    # Split temp into validation and test sets (50% each of remaining 20%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def standardize_data(X_train, X_test, X_val=None, binary_cols=None):
    """Standardize the data (excluding binary columns)."""
    binary_cols = binary_cols or []
    
    for col in X_train.columns:
        if col not in binary_cols:
            mean = X_train[col].mean()
            std = X_train[col].std()
            X_train[col] = (X_train[col] - mean) / std
            X_test[col] = (X_test[col] - mean) / std
            if X_val is not None:
                X_val[col] = (X_val[col] - mean) / std  # Standardize validation set if provided

    return X_train, X_test, X_val

def add_intercept(X):
    """Add a column of ones to the dataset for the intercept."""
    return np.column_stack([np.ones(X.shape[0]), X.values])

def display_statistics(column_names, coefficients, std_errors, z_scores):
    """Display statistics like coefficients, standard errors, and Z-scores."""
    results = pd.DataFrame({
        "Term": column_names,
        "Coefficient": coefficients,
        "Std. Error": std_errors,
        "Z Score": z_scores,
    })
    print(results.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

def plot_correlation_matrix(X_train):
    """Plot a heatmap of the correlation matrix."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    correlation_matrix = X_train.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".3f")
    plt.title("Correlation Matrix (Training Set)")
    plt.show()
