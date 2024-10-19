import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def wrongCV(X, y, selected_predictors=100):

    np.random.seed(42)
    
    correlation = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
    top_predictors = np.argsort(np.abs(correlation))[::-1][:selected_predictors]
    X_selected = X[:, top_predictors]

    
    test_indices = np.random.choice(X.shape[0], 10, replace=False)
    train_indices = np.setdiff1d(np.arange(X.shape[0]), test_indices)

    X_train, X_test = X_selected[train_indices], X_selected[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)

    
    return np.array(
        [np.corrcoef(X_test[:, i], y_test)[0, 1] for i in range(X_test.shape[1])]
    )
