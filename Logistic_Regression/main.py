import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from l1_logistic_regression import L1RegularizedLogisticRegression
from logistic_regression import LogisticRegression
from model_selection import forward_step_selection, select_lambda
from utils import load_and_split_data, logger, plot_feature_importances

# Parameters
DATASET_PATH = "data/south_african_heart_disease.csv"
TARGET_COLUMN = "chd"
TEST_SIZE = 0.1
VAL_SIZE = 0.1
LAMBDA_VALUES = np.logspace(-5, 5, 100)  # Wider range, more values
LEARNING_RATE = 0.05  # Slightly higher learning rate


def main():
    try:
        # Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
            DATASET_PATH, TARGET_COLUMN, TEST_SIZE, VAL_SIZE
        )

        # Preprocess the data
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns
        )
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        logger.info("Data loaded, split, and preprocessed.")
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Validation set shape: {X_val.shape}")
        logger.info(f"Test set shape: {X_test.shape}")

        # Unregularized model
        logger.info("--- Unregularized Logistic Regression ---")
        unregularized_model = LogisticRegression(learning_rate=LEARNING_RATE)
        unregularized_model.fit(X_train_scaled, y_train)
        y_pred_test = unregularized_model.predict(X_test_scaled)
        unregularized_accuracy = accuracy_score(y_test, y_pred_test)
        logger.info(f"Unregularized Model Accuracy: {unregularized_accuracy:.4f}")

        # Stepwise feature selection
        logger.info("--- Stepwise Feature Selection ---")
        selected_features = forward_step_selection(
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val,
            min_improvement=0.001,
            model_class=LogisticRegression,
        )
        logger.info(f"Selected features: {selected_features}")

        stepwise_model = LogisticRegression(learning_rate=LEARNING_RATE)
        stepwise_model.fit(X_train_scaled[selected_features], y_train)
        y_pred_test = stepwise_model.predict(X_test_scaled[selected_features])
        stepwise_accuracy = accuracy_score(y_test, y_pred_test)
        logger.info(f"Stepwise Model Accuracy: {stepwise_accuracy:.4f}")

        # L2 regularized model
        logger.info("--- L2 Regularized Logistic Regression ---")
        best_lambda = select_lambda(
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val,
            LAMBDA_VALUES,
            LogisticRegression,
        )
        logger.info(f"Best lambda: {best_lambda}")

        l2_model = LogisticRegression(
            learning_rate=LEARNING_RATE, lambda_param=best_lambda
        )
        l2_model.fit(X_train_scaled, y_train, regularized=True)
        y_pred_test = l2_model.predict(X_test_scaled)
        l2_accuracy = accuracy_score(y_test, y_pred_test)
        logger.info(f"L2 Regularized Model Accuracy: {l2_accuracy:.4f}")

        # L1 regularized model
        logger.info("--- L1 Regularized Logistic Regression ---")
        best_lambda = select_lambda(
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val,
            LAMBDA_VALUES,
            L1RegularizedLogisticRegression,
        )
        logger.info(f"Best lambda: {best_lambda}")

        l1_model = L1RegularizedLogisticRegression(
            learning_rate=LEARNING_RATE, lambda_param=best_lambda
        )
        l1_model.fit(X_train_scaled, y_train)
        y_pred_test = l1_model.predict(X_test_scaled)
        l1_accuracy = accuracy_score(y_test, y_pred_test)
        logger.info(f"L1 Regularized Model Accuracy: {l1_accuracy:.4f}")

        # Count non-zero features
        non_zero_features = np.sum(l1_model.theta != 0) - 1  # Subtract 1 for bias term
        logger.info(f"Number of non-zero features: {non_zero_features}")

        # Print comparison table
        logger.info("\nModel Comparison:")
        logger.info("Model\t\t\tAccuracy")
        logger.info("---------------------------------")
        logger.info(f"Unregularized\t\t{unregularized_accuracy:.4f}")
        logger.info(f"Stepwise\t\t{stepwise_accuracy:.4f}")
        logger.info(f"L1 Regularized\t\t{l1_accuracy:.4f}")
        logger.info(f"L2 Regularized\t\t{l2_accuracy:.4f}")

        # Plot feature importances
        plot_feature_importances(unregularized_model, X_train.columns)
        plot_feature_importances(l2_model, X_train.columns)
        plot_feature_importances(l1_model, X_train.columns)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
