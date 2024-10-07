import numpy as np
import pandas as pd

from logistic_regression import LogisticRegression
from utils import (evaluate_model, forward_step_selection, load_and_split_data,
                   plot_confusion_matrix, plot_scatterplot_matrix,
                   preprocess_data, select_lambda)


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
    unregularized_model = LogisticRegression(learning_rate=0.1)
    unregularized_model.fit(X_train_scaled.values, y_train.values)
    print("Unregularized model trained.")

    # Make predictions and evaluate the unregularized model
    y_pred_unregularized = unregularized_model.predict(X_test_scaled.values)
    accuracy_unregularized, conf_matrix_unregularized = evaluate_model(
        y_test, y_pred_unregularized
    )

    print("\nUnregularized Model Results:")
    print(f"Accuracy: {accuracy_unregularized:.4f}")
    print("Coefficients:", unregularized_model.theta)
    plot_confusion_matrix(conf_matrix_unregularized)

    # Implement stepwise feature selection
    selected_features = forward_step_selection(X_train_scaled, y_train, X_val_scaled, y_val)
    print(f"Selected features: {selected_features}")

    # Train model with selected features
    stepwise_model = LogisticRegression(learning_rate=0.1)
    stepwise_model.fit(X_train_scaled[selected_features].values, y_train.values)
    print("Stepwise model trained.")

    # Evaluate stepwise model
    y_pred_stepwise = stepwise_model.predict(X_test_scaled[selected_features].values)
    accuracy_stepwise, conf_matrix_stepwise = evaluate_model(
        y_test, y_pred_stepwise
    )

    print("\nStepwise Model Results:")
    print(f"Accuracy: {accuracy_stepwise:.4f}")
    print("Coefficients:", stepwise_model.theta)
    plot_confusion_matrix(conf_matrix_stepwise)

    # Implement L2 regularized logistic regression
    print("\nSelecting best lambda for L2 regularization...")
    best_lambda = select_lambda(X_train_scaled, y_train, X_val_scaled, y_val)
    print(f"Best lambda: {best_lambda}")

    l2_model = LogisticRegression(learning_rate=0.1, lambda_param=best_lambda)
    l2_model.fit_l2(X_train_scaled.values, y_train.values)
    print("L2 model trained.")

    y_pred_l2 = l2_model.predict(X_test_scaled.values)
    accuracy_l2, conf_matrix_l2 = evaluate_model(
        y_test, y_pred_l2
    )

    print("\nL2 Model Results:")
    print(f"Accuracy: {accuracy_l2:.4f}")
    print("Coefficients:", l2_model.theta)
    plot_confusion_matrix(conf_matrix_l2)

    # Print the baseline accuracy (proportion of the majority class)
    baseline_accuracy = max(data["chd"].mean(), 1 - data["chd"].mean())
    print(f"\nBaseline Accuracy: {baseline_accuracy:.4f}")

    # Create a table comparing the accuracies of all models
    print("\nModel Comparison:")
    print("Model\t\t\tAccuracy")
    print("---------------------------------")
    print(f"Baseline\t\t{baseline_accuracy:.4f}")
    print(f"Unregularized\t\t{accuracy_unregularized:.4f}")
    print(f"Stepwise\t\t{accuracy_stepwise:.4f}")
    print(f"L2 Regularized\t\t{accuracy_l2:.4f}")

if __name__ == "__main__":
    main()