# model_selection.py
import numpy as np
from sklearn.metrics import accuracy_score

from utils import logger


def select_lambda(X_train, y_train, X_val, y_val, lambda_values, model_class):
    best_lambda = None
    best_accuracy = -np.inf
    for lambda_param in lambda_values:
        model = model_class(learning_rate=0.01, lambda_param=lambda_param)
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred_val)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lambda = lambda_param
    return best_lambda


def forward_step_selection(
    X_train, y_train, X_val, y_val, model_class, min_improvement=0.001
):
    available_features = list(X_train.columns)
    selected_features = []
    best_accuracy = 0
    improvement_history = []

    while available_features:
        best_feature = None
        best_accuracy_improvement = 0

        for feature in available_features:
            current_features = selected_features + [feature]
            model = model_class(learning_rate=0.01)
            model.fit(X_train[current_features], y_train)

            y_pred_val = model.predict(X_val[current_features])
            accuracy = accuracy_score(y_val, y_pred_val)

            accuracy_improvement = accuracy - best_accuracy
            if accuracy_improvement > best_accuracy_improvement:
                best_accuracy_improvement = accuracy_improvement
                best_feature = feature

        if best_accuracy_improvement > min_improvement:
            selected_features.append(best_feature)
            available_features.remove(best_feature)
            best_accuracy += best_accuracy_improvement
            improvement_history.append((best_feature, best_accuracy))
            logger.info(
                f"Added feature: {best_feature}. New accuracy: {best_accuracy:.4f}"
            )
        else:
            logger.info(f"Stopping: No improvement greater than {min_improvement}")
            break

    logger.info("\nFeature selection history:")
    for feature, accuracy in improvement_history:
        logger.info(f"Feature: {feature}, Accuracy: {accuracy:.4f}")

    return selected_features
