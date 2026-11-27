
from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
PREPROCESSOR_DIR = PROJECT_ROOT / "artifacts" / "preprocessors"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"

SUBSET = "FD001"
RUL_CAP = 125


# ------------------------------------------------------------
# 2. MAIN
# ------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("AEROGUARD AI - PHASE 5.2")
    print("OFFICIAL NASA FD001 TEST EVALUATION")
    print("=" * 70)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 3. LOAD TEST FEATURES AND METADATA
    # --------------------------------------------------------

    print("\n[1/6] Loading endpoint test data...")

    X_test = pd.read_csv(
        DATA_DIR / f"X_test_endpoint_{SUBSET}.csv"
    )

    metadata = pd.read_csv(
        DATA_DIR / f"test_endpoint_metadata_{SUBSET}.csv"
    )

    if len(X_test) != 100 or len(metadata) != 100:
        raise ValueError(
            "Expected exactly 100 endpoint rows and 100 metadata rows."
        )

    if "engine_id" not in metadata.columns:
        raise ValueError(
            "test_endpoint_metadata must contain engine_id."
        )

    metadata = metadata.reset_index(drop=True)

    expected_engine_ids = np.arange(1, 101)

    if not np.array_equal(
        metadata["engine_id"].to_numpy(),
        expected_engine_ids,
    ):
        raise ValueError(
            "Test engine IDs must be 1 through 100 in order."
        )

    print("Test endpoint shape:", X_test.shape)
    print("Test engines:", len(metadata))

    # --------------------------------------------------------
    # 4. LOAD OFFICIAL NASA RUL LABELS
    # --------------------------------------------------------

    print("\n[2/6] Loading official NASA RUL labels...")

    rul_path = RAW_DIR / f"RUL_{SUBSET}.txt"

    y_test_actual = np.loadtxt(
        rul_path,
        dtype=float,
        ndmin=1,
    )

    if len(y_test_actual) != len(X_test):
        raise ValueError(
            "Official RUL label count does not match test engines."
        )

    if not np.isfinite(y_test_actual).all():
        raise ValueError("Official RUL labels contain invalid values.")

    # Official labels correspond to engines 1 through 100.
    y_test_capped = np.minimum(y_test_actual, RUL_CAP)

    print("Official labels:", len(y_test_actual))
    print(
        "Actual RUL range:",
        y_test_actual.min(),
        "to",
        y_test_actual.max(),
    )
    print(
        "Capped RUL range:",
        y_test_capped.min(),
        "to",
        y_test_capped.max(),
    )

    # --------------------------------------------------------
    # 5. LOAD FEATURE COLUMN ORDER
    # --------------------------------------------------------

    print("\n[3/6] Checking feature alignment...")

    feature_columns = joblib.load(
        PREPROCESSOR_DIR / f"feature_columns_{SUBSET}.pkl"
    )

    missing_features = set(feature_columns) - set(X_test.columns)

    if missing_features:
        raise ValueError(
            "Missing test features: "
            f"{sorted(missing_features)}"
        )

    X_test = X_test.reindex(columns=feature_columns)

    if X_test.shape[1] != len(feature_columns):
        raise ValueError("Feature count mismatch.")

    if not np.isfinite(
        X_test.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Test features contain NaN or infinite values."
        )

    print("Expected features:", len(feature_columns))
    print("Test features:", X_test.shape[1])
    print("Feature alignment successful.")

    # --------------------------------------------------------
    # 6. LOAD TRAINED MODELS
    # --------------------------------------------------------

    print("\n[4/6] Loading existing trained models...")

    model_paths = {
        "Random Forest": (
            MODEL_DIR / f"randomforest_{SUBSET}.pkl"
        ),
        "XGBoost": (
            MODEL_DIR / f"xgboost_{SUBSET}.pkl"
        ),
    }

    models = {}

    for model_name, model_path in model_paths.items():

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        models[model_name] = joblib.load(model_path)

        print(f"Loaded: {model_name}")

    # --------------------------------------------------------
    # 7. GENERATE ENDPOINT PREDICTIONS
    # --------------------------------------------------------

    print("\n[5/6] Generating endpoint predictions...")

    predictions = {}

    for model_name, model in models.items():

        predicted_rul = model.predict(X_test)

        predicted_rul = np.asarray(
            predicted_rul,
            dtype=float,
        )

        if not np.isfinite(predicted_rul).all():
            raise ValueError(
                f"{model_name} generated invalid predictions."
            )

        # Same output limits used in Phase 4
        predicted_rul = np.clip(
            predicted_rul,
            0,
            RUL_CAP,
        )

        predictions[model_name] = predicted_rul

        print(
            f"{model_name} prediction range: "
            f"{predicted_rul.min():.2f} "
            f"to {predicted_rul.max():.2f}"
        )

    # --------------------------------------------------------
    # 8. CALCULATE OFFICIAL TEST METRICS
    # --------------------------------------------------------

    print("\n[6/6] Calculating official test metrics...")

    from src.evaluation.metrics import evaluate_regression

    metric_rows = []

    for model_name, predicted_rul in predictions.items():

        # Evaluation against capped official RUL
        capped_metrics = evaluate_regression(
            y_test_capped,
            predicted_rul,
        )

        metric_rows.append({
            "Model": model_name,
            "Evaluation": "Capped RUL",
            **capped_metrics,
        })

        # Evaluation against actual official RUL
        actual_metrics = evaluate_regression(
            y_test_actual,
            predicted_rul,
        )

        metric_rows.append({
            "Model": model_name,
            "Evaluation": "Actual RUL",
            **actual_metrics,
        })

    metrics_df = pd.DataFrame(metric_rows)

    print("\n" + "=" * 70)
    print("OFFICIAL NASA FD001 TEST RESULTS")
    print("=" * 70)

    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------------
    # 9. SAVE ENGINE-LEVEL PREDICTIONS
    # --------------------------------------------------------

    predictions_df = metadata.copy()

    predictions_df["actual_RUL"] = y_test_actual
    predictions_df["capped_RUL"] = y_test_capped

    for model_name, predicted_rul in predictions.items():

        column_name = (
            "RandomForest_prediction"
            if model_name == "Random Forest"
            else "XGBoost_prediction"
        )

        predictions_df[column_name] = predicted_rul

    predictions_path = (
        REPORT_DIR / f"test_predictions_{SUBSET}.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # 10. SAVE METRICS
    # --------------------------------------------------------

    metrics_path = (
        REPORT_DIR / f"official_test_metrics_{SUBSET}.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("PHASE 5.2 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nSaved files:")
    print(f"- {predictions_path}")
    print(f"- {metrics_path}")

    print("\nNo model training or tuning was performed.")


if __name__ == "__main__":
    main()
