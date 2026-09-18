"""Model constructors, identical to notebook 07's `make_model`.

The test fixture models are fitted through this module on frames produced by transform.build_features, so the
fixture carries the same dtypes, encoder maps and estimator settings as the promoted model (Part 0 #15).
The Phase-3 retraining pipeline imports the same functions.
"""
from __future__ import annotations

SEED = 42


def make_model(family: str, params: dict, n_jobs: int = -1):
    if family == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(penalty="l2", solver="lbfgs", max_iter=3000, **params)
    if family == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(objective="binary", subsample_freq=1, random_state=SEED, n_jobs=n_jobs,
                                  deterministic=True, force_row_wise=True, verbose=-1, **params)
    if family == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(objective="binary:logistic", tree_method="hist", enable_categorical=True,
                                 max_cat_to_onehot=1, random_state=SEED, n_jobs=n_jobs, **params)
    raise ValueError(f"unknown family {family!r}")


def fit(family: str, params: dict, X, y, sample_weight=None, n_jobs: int = -1):
    model = make_model(family, params, n_jobs=n_jobs)
    model.fit(X, y, sample_weight=sample_weight)
    return model
