
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "artifacts"
    / "models"
    / "randomforest_FD001.pkl"
)

REPORT_DIR = ROOT / "artifacts" / "reports"
FIGURE_DIR = ROOT / "artifacts" / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. FIND FEATURE COLUMN FILE
# ==================================================

feature_files = list(
    ROOT.rglob("feature_columns_FD001.pkl")
)

if len(feature_files) == 0:
    raise FileNotFoundError(
        "feature_columns_FD001.pkl was not found."
    )

if len(feature_files) > 1:
    raise RuntimeError(
        "Multiple feature column files found. "
        "Please keep only the intended FD001 artifact "
        "or specify its path manually."
    )

FEATURE_COLUMNS_PATH = feature_files[0]


# ==================================================
# 3. LOAD SAVED MODEL AND FEATURE NAMES
# ==================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

feature_columns = joblib.load(
    FEATURE_COLUMNS_PATH
)

feature_columns = list(feature_columns)


# ==================================================
# 4. VALIDATE MODEL
# ==================================================

if not hasattr(model, "feature_importances_"):
    raise TypeError(
        "The loaded model does not support "
        "feature_importances_."
    )

importances = model.feature_importances_

if len(feature_columns) != len(importances):
    raise ValueError(
        f"Feature mismatch: "
        f"{len(feature_columns)} feature names, "
        f"{len(importances)} importance values."
    )


# ==================================================
# 5. CREATE FEATURE IMPORTANCE DATAFRAME
# ==================================================

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": importances,
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False,
).reset_index(drop=True)

importance_df["rank"] = (
    importance_df.index + 1
)

importance_df["importance_percentage"] = (
    importance_df["importance"] * 100
)


# ==================================================
# 6. SAVE COMPLETE RESULTS
# ==================================================

output_csv = (
    REPORT_DIR
    / "feature_importance_FD001.csv"
)

importance_df.to_csv(
    output_csv,
    index=False,
)


# ==================================================
# 7. SELECT TOP 20 FEATURES
# ==================================================

top_features = importance_df.head(20)

print("\n" + "=" * 60)
print("AEROGUARD AI - RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

print(f"\nTotal features: {len(feature_columns)}")

print("\nTOP 20 IMPORTANT FEATURES:\n")

print(
    top_features[
        [
            "rank",
            "feature",
            "importance",
            "importance_percentage",
        ]
    ].round(5).to_string(index=False)
)


# ==================================================
# 8. CREATE FEATURE IMPORTANCE PLOT
# ==================================================

plot_df = top_features.sort_values(
    by="importance",
    ascending=True,
)

plt.figure(figsize=(12, 9))

plt.barh(
    plot_df["feature"],
    plot_df["importance"],
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title(
    "AeroGuard AI - Random Forest "
    "Feature Importance (FD001)"
)

plt.tight_layout()

plot_path = (
    FIGURE_DIR
    / "feature_importance_FD001.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ==================================================
# 9. FEATURE IMPORTANCE SUMMARY
# ==================================================

print("\nOUTPUT FILES:")
print(f"CSV: {output_csv}")
print(f"Plot: {plot_path}")

print("\nFeature importance analysis completed.")
