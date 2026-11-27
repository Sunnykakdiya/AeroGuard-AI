
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


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

REPORT_DIR = ROOT / "artifacts" / "reports"
FIGURE_DIR = ROOT / "artifacts" / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD MODEL AND TEST FEATURES
# ==================================================

print("\nLoading Random Forest model...")

model = joblib.load(MODEL_PATH)

X_test = pd.read_csv(X_PATH)

# Remove an automatically saved CSV index, if present.
X_test = X_test.loc[
    :,
    ~X_test.columns.astype(str).str.startswith("Unnamed:")
]

if X_test.shape[0] != 100:
    raise ValueError(
        f"Expected 100 engines, got {len(X_test)}."
    )

if X_test.shape[1] != model.n_features_in_:
    raise ValueError(
        "Feature count does not match the model."
    )

if not np.isfinite(X_test.to_numpy()).all():
    raise ValueError(
        "Non-finite values found in test features."
    )

print("Test data shape:", X_test.shape)


# ==================================================
# 3. CREATE SHAP EXPLAINER
# ==================================================

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)


# ==================================================
# 4. CALCULATE SHAP VALUES
# ==================================================

print("Calculating SHAP values...")

shap_values = explainer(X_test)

values = np.asarray(shap_values.values)

# This is a single-output regression model.
if values.ndim != 2:
    raise ValueError(
        f"Unexpected SHAP output shape: {values.shape}"
    )

if values.shape != X_test.shape:
    raise ValueError(
        "SHAP values do not match test feature dimensions."
    )

print("SHAP values shape:", values.shape)


# ==================================================
# 5. CREATE GLOBAL SHAP IMPORTANCE
# ==================================================

mean_absolute_shap = np.abs(values).mean(axis=0)

importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "mean_absolute_shap": mean_absolute_shap,
})

importance_df = importance_df.sort_values(
    "mean_absolute_shap",
    ascending=False,
).reset_index(drop=True)

importance_df["rank"] = (
    importance_df.index + 1
)

importance_df = importance_df[
    ["rank", "feature", "mean_absolute_shap"]
]

csv_path = (
    REPORT_DIR / "shap_feature_importance_FD001.csv"
)

importance_df.to_csv(csv_path, index=False)


# ==================================================
# 6. DISPLAY TOP 20 FEATURES
# ==================================================

print("\n" + "=" * 65)
print("AEROGUARD AI - SHAP FEATURE IMPORTANCE")
print("=" * 65)

print("\nTOP 20 FEATURES:\n")

print(
    importance_df.head(20).round(5).to_string(
        index=False
    )
)


# ==================================================
# 7. SHAP SUMMARY / BEESWARM PLOT
# ==================================================

print("\nCreating SHAP summary plot...")

plt.figure()

shap.plots.beeswarm(
    shap_values,
    max_display=20,
    show=False,
)

plt.title(
    "AeroGuard AI - SHAP Summary (FD001)"
)

plt.tight_layout()

beeswarm_path = (
    FIGURE_DIR / "shap_summary_FD001.png"
)

plt.savefig(
    beeswarm_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ==================================================
# 8. SHAP GLOBAL BAR PLOT
# ==================================================

print("Creating SHAP bar plot...")

plt.figure()

shap.plots.bar(
    shap_values,
    max_display=20,
    show=False,
)

plt.title(
    "AeroGuard AI - Mean Absolute SHAP Values"
)

plt.tight_layout()

bar_path = (
    FIGURE_DIR / "shap_bar_FD001.png"
)

plt.savefig(
    bar_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ==================================================
# 9. SAVE SHAP VALUES
# ==================================================

shap_values_df = pd.DataFrame(
    values,
    columns=X_test.columns,
)

shap_values_df.insert(
    0,
    "engine_id",
    np.arange(1, len(X_test) + 1),
)

shap_values_path = (
    REPORT_DIR / "shap_values_FD001.csv"
)

shap_values_df.to_csv(
    shap_values_path,
    index=False,
)


# ==================================================
# 10. COMPLETION
# ==================================================

print("\nOUTPUT FILES:")

print("1.", csv_path)
print("2.", beeswarm_path)
print("3.", bar_path)
print("4.", shap_values_path)

print("\nSHAP analysis completed successfully.")
