"""Fold-fitted VIF selection and name-aware SMOTE for model sensitivity checks."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from imblearn.over_sampling import SMOTE, SMOTENC


class FoldVIFSelector(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=10.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("VIF selection requires named DataFrame columns")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        remaining = list(X.columns)
        self.elimination_log_ = []
        if self.threshold is not None:
            # Constants cannot be used in the auxiliary VIF regressions.
            for col in list(remaining):
                if np.ptp(X[col].to_numpy()) <= 1e-12:
                    remaining.remove(col)
                    self.elimination_log_.append({"feature": col, "reason": "constant"})
            if not remaining:
                raise ValueError("No nonconstant features remain")
            while len(remaining) > 1:
                a = X[remaining].to_numpy(dtype=float)
                a = a - a.mean(axis=0)
                vifs = []
                for j in range(a.shape[1]):
                    target = a[:, j]
                    other = np.delete(a, j, axis=1)
                    residual = target - other @ np.linalg.lstsq(other, target, rcond=None)[0]
                    fraction = float((residual @ residual) / (target @ target))
                    vifs.append(np.inf if fraction <= 1e-12 else 1.0 / fraction)
                j = int(np.argmax(vifs))
                if vifs[j] < self.threshold:
                    break
                col = remaining.pop(j)
                self.elimination_log_.append({"feature": col, "vif": float(vifs[j]), "reason": "vif"})
        self.selected_features_ = remaining
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_features_")
        return X.loc[:, self.selected_features_].copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_features_")
        return np.asarray(self.selected_features_, dtype=object)


class NamedSMOTE(BaseEstimator):
    """Resolve categorical indices after fold-local feature selection."""
    def __init__(self, binary_features=(), random_state=4012):
        self.binary_features = binary_features
        self.random_state = random_state

    def fit_resample(self, X, y):
        idx = [i for i, col in enumerate(X.columns) if col in self.binary_features]
        if len(idx) == X.shape[1]:
            raise ValueError("SMOTE-NC requires at least one continuous feature")
        sampler = (SMOTENC(categorical_features=idx, random_state=self.random_state)
                   if idx else SMOTE(random_state=self.random_state))
        self.sampler_ = sampler
        return sampler.fit_resample(X, y)
