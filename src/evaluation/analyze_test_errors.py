
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "artifacts"
    / "reports"
    / "test_predictions_FD001.csv"
)

REPORT_DIR = ROOT / "artifacts" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. NASA scoring function
# --------------------------------------------------

def nasa_score_per_engine(y_true, y_pred):
    """
    Calculate the NASA C-MAPSS score for each engine.
    Positive error means predicted RUL is too high.
    """
    error = np.asarray(y_pred) - np.asarray(y_true)

    scores = np.where(
        error < 0,
        np.exp(-error / 13.0) - 1.0,
        np.exp(error / 10.0) - 1.0,
    )

    return scores


# --------------------------------------------------
# 3. Load and validate predictions
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "engine_id",
    "actual_RUL",
    "capped_RUL",
    "RandomForest_prediction",
    "XGBoost_prediction",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if df["engine_id"].duplicated().any():
    raise ValueError("Duplicate engine IDs found.")

if len(df) != 100:
    raise ValueError(
        f"Expected 100 test engines, found {len(df)}."
    )


# --------------------------------------------------
# 4. Calculate engine-level errors and NASA scores
# --------------------------------------------------

for model in ["RandomForest", "XGBoost"]:

    prediction_column = f"{model}_prediction"

    # Actual RUL errors
    actual_error = (
        df[prediction_column] - df["actual_RUL"]
    )

    df[f"{model}_actual_error"] = actual_error
    df[f"{model}_actual_absolute_error"] = (
        actual_error.abs()
    )

    df[f"{model}_actual_NASA_score"] = (
        nasa_score_per_engine(
            df["actual_RUL"],
            df[prediction_column],
        )
    )

    # Capped RUL errors
    capped_error = (
        df[prediction_column] - df["capped_RUL"]
    )

    df[f"{model}_capped_error"] = capped_error
    df[f"{model}_capped_absolute_error"] = (
        capped_error.abs()
    )

    df[f"{model}_capped_NASA_score"] = (
        nasa_score_per_engine(
            df["capped_RUL"],
            df[prediction_column],
        )
    )


# --------------------------------------------------
# 5. Model-level summary
# --------------------------------------------------

summary_rows = []

for model in ["RandomForest", "XGBoost"]:

    error = df[f"{model}_capped_error"]

    summary_rows.append({
        "Model": model,
        "Engine_Count": len(df),
        "MAE": error.abs().mean(),
        "RMSE": np.sqrt((error ** 2).mean()),
        "Mean_Bias": error.mean(),
        "Overestimations": (error > 0).sum(),
        "Underestimations": (error < 0).sum(),
        "Exact_Predictions": (error == 0).sum(),
        "NASA_Score": df[
            f"{model}_capped_NASA_score"
        ].sum(),
        "NASA_Mean_Score": df[
            f"{model}_capped_NASA_score"
        ].mean(),
    })

summary = pd.DataFrame(summary_rows)


# --------------------------------------------------
# 6. Worst engines by absolute error
# --------------------------------------------------

worst_rows = []

for model in ["RandomForest", "XGBoost"]:

    error_column = f"{model}_capped_error"
    absolute_column = f"{model}_capped_absolute_error"
    nasa_column = f"{model}_capped_NASA_score"

    worst = df.nlargest(10, absolute_column)

    for _, row in worst.iterrows():

        worst_rows.append({
            "Model": model,
            "Engine_ID": row["engine_id"],
            "Actual_RUL": row["actual_RUL"],
            "Capped_RUL": row["capped_RUL"],
            "Prediction": row[f"{model}_prediction"],
            "Error": row[error_column],
            "Absolute_Error": row[absolute_column],
            "NASA_Score": row[nasa_column],
        })

worst_engines = pd.DataFrame(worst_rows)


# --------------------------------------------------
# 7. Error analysis by actual RUL range
# --------------------------------------------------

df["RUL_Range"] = pd.cut(
    df["actual_RUL"],
    bins=[-1, 25, 50, 75, 100, 125, np.inf],
    labels=[
        "0-25",
        "26-50",
        "51-75",
        "76-100",
        "101-125",
        "Above 125",
    ],
)

range_rows = []

for model in ["RandomForest", "XGBoost"]:

    error_column = f"{model}_capped_error"
    nasa_column = f"{model}_capped_NASA_score"

    for rul_range, group in df.groupby(
        "RUL_Range",
        observed=False,
    ):

        if group.empty:
            continue

        errors = group[error_column]

        range_rows.append({
            "Model": model,
            "RUL_Range": str(rul_range),
            "Engine_Count": len(group),
            "MAE": errors.abs().mean(),
            "RMSE": np.sqrt((errors ** 2).mean()),
            "Mean_Bias": errors.mean(),
            "Overestimations": (errors > 0).sum(),
            "Underestimations": (errors < 0).sum(),
            "NASA_Score": group[nasa_column].sum(),
        })

range_analysis = pd.DataFrame(range_rows)


# --------------------------------------------------
# 8. Save reports
# --------------------------------------------------

df.to_csv(
    REPORT_DIR / "engine_level_errors_FD001.csv",
    index=False,
)

summary.to_csv(
    REPORT_DIR / "engine_error_summary_FD001.csv",
    index=False,
)

worst_engines.to_csv(
    REPORT_DIR / "worst_engines_FD001.csv",
    index=False,
)

range_analysis.to_csv(
    REPORT_DIR / "rul_range_analysis_FD001.csv",
    index=False,
)


# --------------------------------------------------
# 9. Display results
# --------------------------------------------------

print("\n" + "=" * 65)
print("AEROGUARD AI - OFFICIAL TEST ERROR ANALYSIS")
print("=" * 65)

print("\nMODEL SUMMARY (CAPPED RUL)")
print(summary.round(4).to_string(index=False))

print("\nWORST 10 ENGINES PER MODEL")
print(worst_engines.round(3).to_string(index=False))

print("\nERROR BY ACTUAL RUL RANGE")
print(range_analysis.round(3).to_string(index=False))

print("\nReports saved successfully:")
print("1. engine_level_errors_FD001.csv")
print("2. engine_error_summary_FD001.csv")
print("3. worst_engines_FD001.csv")
print("4. rul_range_analysis_FD001.csv")
