
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "artifacts" / "models"
    / "randomforest_FD001.pkl"
)

X_PATH = (
    ROOT / "data" / "processed"
    / "X_test_endpoint_FD001.csv"
)

Y_PATH = (
    ROOT / "data" / "processed"
    / "y_test_FD001.csv"
)

REPORT_DIR = ROOT / "artifacts" / "reports"
FIGURE_DIR = ROOT / "artifacts" / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD MODEL AND TEST DATA
# ==================================================

model = joblib.load(MODEL_PATH)

X_test = pd.read_csv(X_PATH)
y_df = pd.read_csv(Y_PATH)

# Remove an automatically saved CSV index if present.
X_test = X_test.loc[
    :, ~X_test.columns.astype(str).str.startswith("Unnamed:")
]

y_df = y_df.loc[
    :, ~y_df.columns.astype(str).str.startswith("Unnamed:")
]

# Identify the target column.
target_candidates = [
    "RUL_target",
    "RUL",
    "capped_RUL",
    "RUL_actual",
]

target_column = next(
    (
        column for column in target_candidates
        if column in y_df.columns
    ),
    None,
)

if target_column is None:
    if y_df.shape[1] == 1:
        y_test = y_df.iloc[:, 0].to_numpy()
    else:
        raise ValueError(
            "Could not identify the RUL target column. "
            f"Available columns: {list(y_df.columns)}"
        )
else:
    y_test = y_df[target_column].to_numpy()


# ==================================================
# 3. VALIDATE TEST DATA
# ==================================================

if len(X_test) != len(y_test):
    raise ValueError(
        "X_test and y_test have different row counts."
    )

if X_test.shape[1] != model.n_features_in_:
    raise ValueError(
        f"Expected {model.n_features_in_} features, "
        f"but received {X_test.shape[1]}."
    )

if len(X_test) != 100:
    raise ValueError(
        f"Expected 100 endpoint rows, got {len(X_test)}."
    )

if not np.isfinite(X_test.to_numpy()).all():
    raise ValueError("Non-finite values found in X_test.")

if not np.isfinite(y_test).all():
    raise ValueError("Non-finite values found in y_test.")


# ==================================================
# 4. CALCULATE BASELINE MAE
# ==================================================

baseline_predictions = np.clip(
    model.predict(X_test),
    0,
    125,
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions,
)

print("\n" + "=" * 65)
print("AEROGUARD AI - PERMUTATION IMPORTANCE")
print("=" * 65)

print(f"\nTest engines: {len(X_test)}")
print(f"Features: {X_test.shape[1]}")
print(f"Baseline MAE: {baseline_mae:.4f}")


# ==================================================
# 5. CALCULATE PERMUTATION IMPORTANCE
# ==================================================

def capped_mae_scorer(estimator, X, y):
    predictions = np.clip(
        estimator.predict(X),
        0,
        125,
    )

    return -mean_absolute_error(y, predictions)


print("\nCalculating permutation importance...")

result = permutation_importance(
    estimator=model,
    X=X_test,
    y=y_test,
    scoring=capped_mae_scorer,
    n_repeats=20,
    random_state=42,
    n_jobs=-1,
)


# ==================================================
# 6. CREATE RESULTS DATAFRAME
# ==================================================

importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std,
})

importance_df = importance_df.sort_values(
    "importance_mean",
    ascending=False,
).reset_index(drop=True)

importance_df["rank"] = importance_df.index + 1

importance_df = importance_df[
    [
        "rank",
        "feature",
        "importance_mean",
        "importance_std",
    ]
]


# ==================================================
# 7. SAVE COMPLETE RESULTS
# ==================================================

csv_path = (
    REPORT_DIR
    / "permutation_importance_FD001.csv"
)

importance_df.to_csv(csv_path, index=False)


# ==================================================
# 8. DISPLAY TOP 20 FEATURES
# ==================================================

print("\nTOP 20 PERMUTATION IMPORTANT FEATURES:\n")

print(
    importance_df.head(20).round(5).to_string(
        index=False
    )
)


# ==================================================
# 9. PLOT TOP 20 FEATURES
# ==================================================

top_features = importance_df.head(20).copy()

plot_df = top_features.sort_values(
    "importance_mean",
    ascending=True,
)

plt.figure(figsize=(12, 9))

plt.barh(
    plot_df["feature"],
    plot_df["importance_mean"],
    xerr=plot_df["importance_std"],
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=1,
)

plt.xlabel("Increase in MAE after shuffling")
plt.ylabel("Feature")

plt.title(
    "AeroGuard AI - Permutation Importance (FD001)"
)

plt.tight_layout()

plot_path = (
    FIGURE_DIR
    / "permutation_importance_FD001.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ==================================================
# 10. COMPLETION
# ==================================================

print("\nBASELINE MAE:", round(baseline_mae, 4))

print("\nOUTPUT FILES:")
print(csv_path)
print(plot_path)

print("\nPermutation importance completed.")
