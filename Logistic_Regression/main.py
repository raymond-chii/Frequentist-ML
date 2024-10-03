import numpy as np
import pandas as pd

from logistic_regression import LogisticRegression
from utils import (
    evaluate_model,
    load_and_split_data,
    plot_confusion_matrix,
    plot_scatterplot_matrix,
    preprocess_data,
)


def main():
    # Load and split the data
    filepath = "data/south_african_heart_disease.csv"
    target_col = "chd"
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
        filepath, target_col
    )

    print("Data loaded and split.")
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Plot the scatterplot matrix
    data = pd.read_csv(filepath)
    plot_scatterplot_matrix(data, target_col)

    # Preprocess the data
    X_train_scaled, X_val_scaled, X_test_scaled = preprocess_data(
        X_train, X_val, X_test
    )
    print("Data preprocessed and scaled.")

    # Train the unregularized model
    unregularized_model = LogisticRegression(learning_rate=0.01)
    unregularized_model.fit(X_train_scaled.values, y_train.values)
    print("Unregularized model trained.")

    # Make predictions and evaluate the unregularized model
    y_pred_unregularized = unregularized_model.predict(X_test_scaled.values)
    accuracy_unregularized, conf_matrix_unregularized = evaluate_model(
        y_test, y_pred_unregularized
    )

    print("\nUnregularized Model Results:")
    print(f"Accuracy: {accuracy_unregularized:.4f}")
    plot_confusion_matrix(conf_matrix_unregularized)

    # TODO: Implement stepwise feature selection

    # TODO: Implement L2 regularized logistic regression

    # Print the baseline accuracy (proportion of the majority class)
    baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())
    print(f"\nBaseline Accuracy: {baseline_accuracy:.4f}")

    # TODO: Create a table comparing the accuracies of all models


if __name__ == "__main__":
    main()
