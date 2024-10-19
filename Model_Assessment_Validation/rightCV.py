import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def rightCV(X, y, selected_predictors=100):

    np.random.seed(42)
    
    test_indices = np.random.choice(X.shape[0], 10, replace=False)
    train_indices = np.setdiff1d(np.arange(X.shape[0]), test_indices)

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    correlation = np.array(
        [np.corrcoef(X_train[:, i], y_train)[0, 1] for i in range(X_train.shape[1])]
    )
    top_predictors = np.argsort(np.abs(correlation))[::-1][:selected_predictors]

    X_train_selected = X_train[:, top_predictors]
    X_test_selected = X_test[:, top_predictors]

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train_selected, y_train)

    return np.array(
        [
            np.corrcoef(X_test_selected[:, i], y_test)[0, 1]
            for i in range(X_test_selected.shape[1])
        ]
    )
