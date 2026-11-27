from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cv import create_engine_cv
from src.models.random_forest import (
    create_random_forest_search,
)
from src.models.xgboost_model import (
    create_xgboost_search,
)
from src.evaluation.metrics import evaluate_regression


SUBSET = "FD001"
RANDOM_STATE = 42
N_SPLITS = 5
N_ITER = 20
RUL_CAP = 125

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"


def load_training_artifacts():

    X_train = pd.read_csv(
        DATA_DIR / f"X_train_{SUBSET}.csv"
    )

    X_val = pd.read_csv(
        DATA_DIR / f"X_val_{SUBSET}.csv"
    )

    y_train = pd.read_csv(
        DATA_DIR / f"y_train_{SUBSET}.csv"
    )["RUL_target"]

    y_val = pd.read_csv(
        DATA_DIR / f"y_val_{SUBSET}.csv"
    )["RUL_target"]

    train_metadata = pd.read_csv(
        DATA_DIR / f"train_metadata_{SUBSET}.csv"
    )

    val_metadata = pd.read_csv(
        DATA_DIR / f"val_metadata_{SUBSET}.csv"
    )

    return (
        X_train,
        X_val,
        y_train,
        y_val,
        train_metadata,
        val_metadata,
    )


def train_and_evaluate(
    model_name,
    search,
    X_train,
    y_train,
    groups,
    X_val,
    y_val,
):
    print("\n" + "=" * 60)
    print(f"TUNING {model_name}")
    print("=" * 60)

    # Group-aware cross-validation
    search.fit(
        X_train,
        y_train,
        groups=groups,
    )

    best_model = search.best_estimator_

    print("\nBest CV MAE:")
    print(-search.best_score_)

    print("\nBest parameters:")
    print(search.best_params_)

    # Predict validation data
    predictions = best_model.predict(X_val)

    # Since the training target was capped,
    # predictions should respect the same target range.
    predictions = np.clip(
        predictions,
        0,
        RUL_CAP,
    )

    metrics = evaluate_regression(
        y_val,
        predictions,
    )

    print("\nValidation metrics:")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return {
        "model": best_model,
        "search": search,
        "predictions": predictions,
        "metrics": metrics,
        "best_params": search.best_params_,
        "best_cv_mae": float(-search.best_score_),
    }


def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Phase 3 artifacts...")

    (
        X_train,
        X_val,
        y_train,
        y_val,
        train_metadata,
        val_metadata,
    ) = load_training_artifacts()

    groups = train_metadata["engine_id"]

    # Safety checks
    assert len(X_train) == len(y_train) == len(groups)
    assert len(X_val) == len(y_val)

    assert list(X_train.columns) == list(X_val.columns)

    train_engines = set(groups)
    val_engines = set(val_metadata["engine_id"])

    if train_engines.intersection(val_engines):
        raise ValueError(
            "Engine leakage between train and validation!"
        )

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Training engines:", len(train_engines))
    print("Validation engines:", len(val_engines))
    print("Features:", X_train.shape[1])

    # Grouped CV
    cv = create_engine_cv(
        n_splits=N_SPLITS,
    )

    results = {}

    # Random Forest
    rf_search = create_random_forest_search(
        cv=cv,
        random_state=RANDOM_STATE,
        n_iter=N_ITER,
    )

    results["RandomForest"] = train_and_evaluate(
        model_name="Random Forest",
        search=rf_search,
        X_train=X_train,
        y_train=y_train,
        groups=groups,
        X_val=X_val,
        y_val=y_val,
    )

    # XGBoost
    xgb_search = create_xgboost_search(
        cv=cv,
        random_state=RANDOM_STATE,
        n_iter=N_ITER,
    )

    results["XGBoost"] = train_and_evaluate(
        model_name="XGBoost",
        search=xgb_search,
        X_train=X_train,
        y_train=y_train,
        groups=groups,
        X_val=X_val,
        y_val=y_val,
    )

    # Comparison table
    comparison = pd.DataFrame({
        name: result["metrics"]
        for name, result in results.items()
    }).T

    comparison["CV_MAE"] = {
        name: result["best_cv_mae"]
        for name, result in results.items()
    }

    comparison.to_csv(
        REPORT_DIR / f"model_comparison_{SUBSET}.csv"
    )

    print("\nMODEL COMPARISON")
    print(comparison.sort_values("MAE"))

    # Select best model based on MAE
    best_name = comparison["MAE"].idxmin()

    best_model = results[best_name]["model"]

    print("\nBest model:", best_name)


    # Evaluate against the capped training target
    capped_metrics = evaluate_regression(
        y_val,
        results[best_name]["predictions"],
    )

    # Evaluate against the actual uncapped remaining life
    actual_rul = val_metadata["RUL"].to_numpy()

    actual_life_metrics = evaluate_regression(
        actual_rul,
        results[best_name]["predictions"],
    )

    print("\nCAPPED TARGET EVALUATION:")
    print(capped_metrics)

    print("\nACTUAL REMAINING LIFE EVALUATION:")
    print(actual_life_metrics)


    # Save all models
    for name, result in results.items():

        filename = (
            name.lower().replace(" ", "_")
            + f"_{SUBSET}.pkl"
        )

        joblib.dump(
            result["model"],
            MODEL_DIR / filename,
        )

    joblib.dump(
        best_model,
        MODEL_DIR / f"best_model_{SUBSET}.pkl",
    )

    # Save predictions
    prediction_df = val_metadata.copy()

    prediction_df["actual_RUL_target"] = y_val.to_numpy()

    for name, result in results.items():
        prediction_df[
            f"{name}_prediction"
        ] = result["predictions"]

    prediction_df.to_csv(
        REPORT_DIR / f"predictions_{SUBSET}.csv",
        index=False,
    )

    # Feature importance
    if hasattr(best_model, "feature_importances_"):

        importance_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": best_model.feature_importances_,
        })

        importance_df = importance_df.sort_values(
            "importance",
            ascending=False,
        )

        importance_df.to_csv(
            REPORT_DIR / f"feature_importance_{SUBSET}.csv",
            index=False,
        )

    # Save metrics
    metrics_json = {
        "dataset": SUBSET,
        "best_model": best_name,
        "rul_cap": RUL_CAP,
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "n_iter": N_ITER,
        "feature_count": X_train.shape[1],
        "training_engines": len(train_engines),
        "validation_engines": len(val_engines),
        "models": {
            name: {
                "metrics": result["metrics"],
                "best_params": result["best_params"],
                "best_cv_mae": result["best_cv_mae"],
            }
            for name, result in results.items()
        },
    }

    with open(
        REPORT_DIR / f"model_metrics_{SUBSET}.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics_json,
            file,
            indent=4,
        )

    print("\nModel artifacts saved successfully.")


if __name__ == "__main__":
    main()
