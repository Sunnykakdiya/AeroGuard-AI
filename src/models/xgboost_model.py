from __future__ import annotations

from scipy.stats import randint, uniform

from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV


def create_xgboost_search(
    cv,
    random_state: int = 42,
    n_iter: int = 20,
):
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
    )

    param_distributions = {
        "n_estimators": [300, 500, 700, 1000],
        "max_depth": randint(2, 9),
        "learning_rate": uniform(0.01, 0.19),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 10),
        "reg_alpha": [0, 0.001, 0.01, 0.1, 1.0],
        "reg_lambda": [0.1, 1.0, 5.0, 10.0],
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
