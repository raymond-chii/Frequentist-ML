# lasso_regression.py
from sklearn.linear_model import LassoCV


class LassoRegression:
    def __init__(self, cv=5, random_state=42):
        self.model = LassoCV(cv=cv, random_state=random_state)
        self.best_alpha = None

    def fit(self, X, y):
        self.model.fit(X, y)
        self.best_alpha = self.model.alpha_

    def predict(self, X):
        return self.model.predict(X)

    @property
    def coefficients(self):  # @property decorator
        return self.model.coef_
