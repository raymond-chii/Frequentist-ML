import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logistic_regression import LogisticRegression


def load_and_split_data(filepath, target_col, test_size=0.1, val_size=0.1):
    data = pd.read_csv(filepath)
    y = data[target_col]
    X = data.drop([target_col], axis=1)

    # Handle categorical variables
    if "famhist" in X.columns:
        X = pd.get_dummies(X, columns=["famhist"], drop_first=True)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_scaled, X_val_scaled, X_test_scaled


def plot_scatterplot_matrix(data, target_col):

    # variable = ["sbp", "tobacco", "ldl", "famhist", "obesity", "alcohol", "age"]
    # Handle categorical variables
    if "famhist" in data.columns:
        data_encoded = pd.get_dummies(data, columns=["famhist"], drop_first=True)

    plt.figure(figsize=(20, 20))
    g = sns.PairGrid(data_encoded, hue=target_col)
    g.map_diag(sns.histplot, hue=None, color=".3")
    g.map_offdiag(sns.scatterplot)

    plt.suptitle(
        "Figure 4.12: Scatterplot Matrix of South African Heart Disease Data", y=1.02
    )
    plt.tight_layout()
    plt.savefig("results/scatterplot.pdf")


def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return accuracy, conf_matrix


def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def forward_step_selection(X_train, y_train, X_val, y_val):
    features = list(X_train.columns)
    selected_features = []
    best_accuracy = 0

    for _ in range(len(features)):
        best_feature = None
        best_accuracy_improvement = 0

        for feature in features:
            if feature not in selected_features:
                current_features = selected_features + [feature]
                model = LogisticRegression(learning_rate=0.01)
                model.fit(X_train[current_features].values, y_train.values)

                y_pred_val = model.predict(X_val[current_features].values)
                accuracy, _ = evaluate_model(y_val, y_pred_val)

                accuracy_improvement = accuracy - best_accuracy
                if accuracy_improvement > best_accuracy_improvement:
                    best_accuracy_improvement = accuracy_improvement
                    best_feature = feature

        if best_feature is not None:
            selected_features.append(best_feature)
            best_accuracy += best_accuracy_improvement
            print(f"Added feature: {best_feature}. New accuracy: {best_accuracy:.4f}")
        else:
            break

    return selected_features

def select_lambda(X_train, y_train, X_val, y_val):
    lambda_values = [0.001, 0.01, 0.1, 1, 10, 100]

    best_lambda = None
    best_accuracy = 0
    for lambda_param in lambda_values:
        model = LogisticRegression(learning_rate=0.1, lambda_param=lambda_param)
        model.fit_l2(X_train.values, y_train.values)
        y_val_pred = model.predict(X_val.values)
        accuracy = accuracy_score(y_val, y_val_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lambda = lambda_param

    return best_lambda

