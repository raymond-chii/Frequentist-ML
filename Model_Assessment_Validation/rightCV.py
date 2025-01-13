import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def rightCV(X, y, n_predictors=100, n_test=10):
    
    test_indices = np.random.choice(X.shape[0], n_test, replace=False)
    train_indices = np.setdiff1d(np.arange(X.shape[0]), test_indices)
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    correlations = np.array([np.corrcoef(X_train[:, i], y_train)[0, 1] 
                           for i in range(X_train.shape[1])])
    top_predictors = np.argsort(abs(correlations))[-n_predictors:]
    
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train[:, top_predictors], y_train)
    
    y_pred = knn.predict(X_test[:, top_predictors])
    error_rate = np.mean(y_pred != y_test)
    
    test_correlations = [np.corrcoef(X_test[:, pred], y_test)[0, 1] 
                        for pred in top_predictors]
    
    print(f"Right way error rate: {error_rate:.3f}")
    return test_correlations