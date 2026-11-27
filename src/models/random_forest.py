from __future__ import annotations

from scipy.stats import randint, uniform

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


def create_random_forest_search(
    cv,
    random_state: int = 42,
    n_iter: int = 20,
):
    model = RandomForestRegressor(
        random_state=random_state,
        n_jobs=1,
    )

    param_distributions = {
        "n_estimators": [200, 300, 500, 700],
        "max_depth": [None, 10, 15, 20, 30],
        "min_samples_split": randint(2, 15),
        "min_samples_leaf": randint(1, 8),
        "max_features": [
            0.5,
            0.7,
            0.9,
            1.0,
            "sqrt",
        ],
        "max_samples": [
            None,
            0.7,
            0.85,
        ],
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
        refit=True,
    )

    return search
