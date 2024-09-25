import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class LinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        return X @ self.coefficients


# Load and prepare data
data = pd.read_csv("prostate_cancer.csv")
y = data["lpsa"]
X = data.drop(["lpsa", "train"], axis=1)

# Split into train and test
train_mask = data["train"] == "T"
X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[~train_mask]
y_test = y[~train_mask]

# Standardize non-binary predictors using only training data
binary_cols = ["svi"]
for col in X_train.columns:
    if col not in binary_cols:
        mean = X_train[col].mean()
        std = X_train[col].std()
        X_train[col] = (X_train[col] - mean) / std
        X_test[col] = (X_test[col] - mean) / std  # Use train mean and std for test set


column_names = list(X_train.columns)

# Calculate correlation matrix for training set
correlation_matrix = X_train.corr()

# Add constant term for intercept
X_train_with_intercept = np.column_stack([np.ones(X_train.shape[0]), X_train.values])
X_test_with_intercept = np.column_stack([np.ones(X_test.shape[0]), X_test.values])

# Update column names to include intercept
column_names = ["Intercept"] + column_names

# Train model on training set
model = LinearRegression()
model.fit(X_train_with_intercept, y_train)

# Calculate residuals and standard errors
y_train_hat = model.predict(X_train_with_intercept)
residuals = y_train - y_train_hat
n, p = X_train_with_intercept.shape
sigma_squared = np.sum(residuals**2) / (n - p)
var_coeffs = sigma_squared * np.linalg.inv(
    X_train_with_intercept.T @ X_train_with_intercept
)
std_errors = np.sqrt(np.diag(var_coeffs))

# Calculate Z scores
z_scores = model.coefficients / std_errors

# Create Table 3.2
results = pd.DataFrame(
    {
        "Term": column_names,
        "Coefficient": model.coefficients,
        "Std. Error": std_errors,
        "Z Score": z_scores,
    }
)

print("Table 3.2: Linear model fit (using training set)")
print(results.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# Plot correlation matrix (Table 3.1)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".3f")
plt.title("Table 3.1: Correlations of predictors (training set)")
plt.show()

# Calculate MSE on test set
y_test_hat = model.predict(X_test_with_intercept)
mse = np.mean((y_test - y_test_hat) ** 2)
print(f"\nMean Squared Error on test set: {mse:.4f}")
