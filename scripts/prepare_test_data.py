
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from pathlib import Path
import sys


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PREPROCESSOR_DIR = PROJECT_ROOT / "artifacts" / "preprocessors"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"

SUBSET = "FD001"
RUL_CAP = 125

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.engineering import (
    create_temporal_features,
    validate_engineered_features,
)


# ============================================================
# 2. LOAD OFFICIAL NASA TEST DATA
# ============================================================

def load_test_data() -> pd.DataFrame:
    """Load the official NASA C-MAPSS FD001 test file."""

    columns = [
        "engine_id",
        "cycle",
        "setting_1",
        "setting_2",
        "setting_3",
    ] + [
        f"sensor_{i}" for i in range(1, 22)
    ]

    file_path = DATA_DIR / f"test_{SUBSET}.txt"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=columns,
        engine="python",
    )

    return df


def load_test_rul() -> pd.Series:
    """Load official RUL labels for the test engines."""

    file_path = DATA_DIR / f"RUL_{SUBSET}.txt"

    if not file_path.exists():
        raise FileNotFoundError(
            f"RUL labels not found: {file_path}"
        )

    rul = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        engine="python",
    ).iloc[:, 0]

    return rul.astype(int).rename("RUL_actual")


# ============================================================
# 3. LOAD SAVED PHASE 3 PREPROCESSING ARTIFACTS
# ============================================================

def load_preprocessing_artifacts():

    sensors_path = (
        PREPROCESSOR_DIR /
        f"selected_sensors_{SUBSET}.pkl"
    )

    features_path = (
        PREPROCESSOR_DIR /
        f"feature_columns_{SUBSET}.pkl"
    )

    if not sensors_path.exists():
        raise FileNotFoundError(
            f"Selected sensors artifact not found: {sensors_path}"
        )

    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature columns artifact not found: {features_path}"
        )

    selected_sensors = joblib.load(sensors_path)
    feature_columns = joblib.load(features_path)

    return selected_sensors, feature_columns


# ============================================================
# 4. GENERATE TEMPORAL FEATURES
# ============================================================

def generate_test_features(
    test_df: pd.DataFrame,
    selected_sensors: list[str],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:

    print("\nGenerating temporal features...")

    engineered_df = create_temporal_features(
        test_df,
        selected_sensors=selected_sensors,
    )

    validate_engineered_features(
        engineered_df,
        feature_columns,
    )

    # Keep only the final observed cycle for each engine.
    endpoint_df = (
        engineered_df
        .sort_values(["engine_id", "cycle"])
        .groupby("engine_id", sort=True)
        .tail(1)
        .sort_values("engine_id")
        .reset_index(drop=True)
    )

    # Preserve engine-level endpoint information.
    endpoint_metadata = endpoint_df[
        ["engine_id", "cycle"]
    ].copy()

    # Match the exact feature names and order used in Phase 4.
    X_test = endpoint_df[
        feature_columns
    ].copy()

    if list(X_test.columns) != feature_columns:
        raise ValueError(
            "Test feature columns do not match saved feature columns."
        )

    values = X_test.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Test features contain NaN or infinity."
        )

    return X_test, endpoint_metadata


# ============================================================
# 5. VALIDATE AND SAVE TEST DATA
# ============================================================

def main():

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("AEROGUARD AI - PHASE 5.1")
    print("OFFICIAL NASA FD001 TEST DATA PREPARATION")
    print("=" * 65)

    # Load official test data.
    print("\n[1/5] Loading official test data...")

    test_df = load_test_data()
    test_rul = load_test_rul()

    print("Test rows:", len(test_df))
    print("Test engines:", test_df["engine_id"].nunique())
    print("Official RUL labels:", len(test_rul))

    # Load saved Phase 3 configuration.
    print("\n[2/5] Loading saved preprocessing artifacts...")

    selected_sensors, feature_columns = (
        load_preprocessing_artifacts()
    )

    print("Selected sensors:", len(selected_sensors))
    print("Expected feature count:", len(feature_columns))

    # Validate engine and label counts.
    engine_ids = sorted(test_df["engine_id"].unique())

    if len(engine_ids) != len(test_rul):
        raise ValueError(
            "Number of test engines does not match "
            "the number of official RUL labels."
        )

    if engine_ids != list(range(1, len(test_rul) + 1)):
        raise ValueError(
            "Unexpected test engine IDs or ordering."
        )

    # Generate endpoint features.
    print("\n[3/5] Generating endpoint features...")

    X_test, endpoint_metadata = generate_test_features(
        test_df,
        selected_sensors,
        feature_columns,
    )

    # Official RUL labels correspond to engine IDs in ascending order.
    endpoint_metadata["RUL_actual"] = (
        test_rul.to_numpy()
    )

    endpoint_metadata["RUL_target"] = (
        endpoint_metadata["RUL_actual"].clip(
            upper=RUL_CAP
        )
    )

    # Confirm endpoint alignment.
    if len(X_test) != len(test_rul):
        raise ValueError(
            "Endpoint feature count does not match RUL label count."
        )

    if not np.array_equal(
        endpoint_metadata["engine_id"].to_numpy(),
        np.asarray(engine_ids),
    ):
        raise ValueError(
            "Endpoint engine order does not match official RUL order."
        )

    # Save endpoint features.
    print("\n[4/5] Saving test artifacts...")

    X_test.to_csv(
        PROCESSED_DIR / f"X_test_endpoint_{SUBSET}.csv",
        index=False,
    )

    endpoint_metadata.to_csv(
        PROCESSED_DIR / f"test_endpoint_metadata_{SUBSET}.csv",
        index=False,
    )

    endpoint_metadata[["RUL_target"]].to_csv(
        PROCESSED_DIR / f"y_test_{SUBSET}.csv",
        index=False,
    )

    # Save a combined reference report.
    endpoint_report = pd.concat(
        [
            endpoint_metadata.reset_index(drop=True),
            X_test.reset_index(drop=True),
        ],
        axis=1,
    )

    endpoint_report.to_csv(
        REPORT_DIR / f"test_endpoint_features_{SUBSET}.csv",
        index=False,
    )

    # Final summary.
    print("\n[5/5] Validation summary...")

    print("\n" + "=" * 65)
    print("PHASE 5.1 COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print("Test engines:", len(X_test))
    print("Endpoint feature shape:", X_test.shape)
    print("Feature count:", X_test.shape[1])

    print(
        "Actual RUL range:",
        int(endpoint_metadata["RUL_actual"].min()),
        "to",
        int(endpoint_metadata["RUL_actual"].max()),
    )

    print(
        "Capped RUL range:",
        int(endpoint_metadata["RUL_target"].min()),
        "to",
        int(endpoint_metadata["RUL_target"].max()),
    )

    print("\nSaved files:")
    print(f"- {PROCESSED_DIR / f'X_test_endpoint_{SUBSET}.csv'}")
    print(f"- {PROCESSED_DIR / f'test_endpoint_metadata_{SUBSET}.csv'}")
    print(f"- {PROCESSED_DIR / f'y_test_{SUBSET}.csv'}")
    print(f"- {REPORT_DIR / f'test_endpoint_features_{SUBSET}.csv'}")


if __name__ == "__main__":
    main()
