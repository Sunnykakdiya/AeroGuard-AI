
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "reports"
)

SUBSET = "FD001"


def main():

    importance_path = (
        REPORT_DIR
        / f"feature_importance_{SUBSET}.csv"
    )

    importance = pd.read_csv(
        importance_path
    )

    top_features = importance.head(20)

    plt.figure(figsize=(12, 8))

    sns.barplot(
        data=top_features,
        x="importance",
        y="feature",
    )

    plt.title(
        "Top 20 Features for RUL Prediction"
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")

    plt.tight_layout()

    output_path = (
        REPORT_DIR
        / f"feature_importance_plot_{SUBSET}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print("Saved:", output_path)


    

    prediction_df = pd.read_csv(
        "artifacts/reports/predictions_FD001.csv"
    )

    model_column = "XGBoost_prediction"

    plt.figure(figsize=(8, 8))

    plt.scatter(
        prediction_df["actual_RUL_target"],
        prediction_df[model_column],
        alpha=0.3,
    )

    max_value = 125

    plt.plot(
        [0, max_value],
        [0, max_value],
        linestyle="--",
    )

    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Actual vs Predicted RUL")

    plt.xlim(0, max_value)
    plt.ylim(0, max_value)

    plt.grid(alpha=0.3)

    plt.show()


    prediction_df["residual"] = (
        prediction_df["actual_RUL_target"]
        -prediction_df["XGBoost_prediction"]
    )

    plt.figure(figsize=(10, 5))

    plt.scatter(
        prediction_df["actual_RUL_target"],
        prediction_df["residual"],
        alpha=0.3,
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.xlabel("Actual RUL")
    plt.ylabel("Residual")
    plt.title("RUL Prediction Residuals")

    plt.grid(alpha=0.3)

    plt.show()

    

if __name__ == "__main__":
    main()
