import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

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
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Perform Lasso regression with cross-validation
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)

# Get the best alpha (lambda) value
best_alpha = lasso_cv.alpha_

# Train final model with best alpha
lasso_best = LassoCV(alphas=[best_alpha], cv=5, random_state=42)
lasso_best.fit(X_train_scaled, y_train)

# Evaluate on test set
y_test_pred = lasso_best.predict(X_test_scaled)
test_mse = np.mean((y_test - y_test_pred) ** 2)

print(f"Best alpha (lambda): {best_alpha}")
print(f"Test MSE: {test_mse}")

# Plot Lasso coefficients
def plot_lasso_coefficients(alphas, coefs, feature_names):
    plt.figure(figsize=(8, 8))
    
    # Calculate shrinkage factor s
    s_values = np.sum(np.abs(coefs), axis=1) / np.sum(np.abs(coefs[0]))
    
    for i, feature in enumerate(feature_names):
        plt.plot(s_values, coefs[:, i], label=feature)
    
    plt.xlabel('Shrinkage Factor s')
    plt.ylabel('Coefficients')
    plt.title('Lasso Coefficient Profiles')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Add vertical line at the chosen lambda
    best_s = np.sum(np.abs(lasso_best.coef_)) / np.sum(np.abs(coefs[0]))
    plt.axvline(x=best_s, color='r', linestyle='--', alpha=0.5, label='Chosen λ')
    
    # plt.xscale('log')
    plt.xlim(0, 1)
    # plt.xticks(range(0, 1, 0.2))
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()

# Generate a range of alpha values
alphas = np.logspace(-4, 0, 100)

# Compute Lasso coefficients for each alpha
coefs = []
for alpha in alphas:
    lasso = LassoCV(alphas=[alpha], cv=5, random_state=42)
    lasso.fit(X_train_scaled, y_train)
    coefs.append(lasso.coef_)

coefs = np.array(coefs)

# Plot Lasso coefficients
plot_lasso_coefficients(alphas, coefs, feature_names)
