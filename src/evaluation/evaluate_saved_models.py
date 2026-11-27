
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_regression


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD VALIDATION DATA
# ============================================================

X_val = pd.read_csv(DATA_DIR / "X_val_FD001.csv")

y_val_capped = pd.read_csv(
    DATA_DIR / "y_val_FD001.csv"
).squeeze("columns")

val_metadata = pd.read_csv(
    DATA_DIR / "val_metadata_FD001.csv"
)

y_val_actual = val_metadata["RUL"]


# ============================================================
# 3. VALIDATE DATA
# ============================================================

if not (
    len(X_val)
    == len(y_val_capped)
    == len(y_val_actual)
):
    raise ValueError("Validation data lengths do not match.")

if y_val_actual.isna().any():
    raise ValueError("Actual RUL contains missing values.")


# ============================================================
# 4. LOAD EXISTING TRAINED MODELS
# ============================================================

model_paths = {
    "Random Forest": MODEL_DIR / "randomforest_FD001.pkl",
    "XGBoost": MODEL_DIR / "xgboost_FD001.pkl",
}

models = {}

for name, path in model_paths.items():

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    models[name] = joblib.load(path)

    print(f"Loaded {name} successfully.")


# ============================================================
# 5. EVALUATE CAPPED AND ACTUAL RUL
# ============================================================

results = []

for model_name, model in models.items():

    print(f"\nEvaluating {model_name}...")

    # Predict using the saved trained model
    predictions = model.predict(X_val)

    # Apply the same prediction limits used in Phase 4
    predictions = np.clip(predictions, 0, 125)

    # Evaluation against capped RUL
    capped_metrics = evaluate_regression(
        y_val_capped,
        predictions
    )

    # Evaluation against actual RUL
    actual_metrics = evaluate_regression(
        y_val_actual,
        predictions
    )

    results.append({
        "Model": model_name,
        "Evaluation": "Capped RUL",
        **capped_metrics,
    })

    results.append({
        "Model": model_name,
        "Evaluation": "Actual RUL",
        **actual_metrics,
    })


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("AEROGUARD AI — PART K: CAPPED VS ACTUAL RUL")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 7. SAVE RESULTS
# ============================================================

output_path = (
    REPORT_DIR / "capped_vs_actual_metrics_FD001.csv"
)

results_df.to_csv(output_path, index=False)

print("\nEvaluation completed successfully.")
print(f"Results saved to: {output_path}")
