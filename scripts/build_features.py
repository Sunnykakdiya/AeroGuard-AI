
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingestion import load_train_data
from src.data.metadata import add_training_rul
from src.features.splitting import split_by_engine
from src.features.selection import select_variable_sensors
from src.features.engineering import (
    create_temporal_features,
    validate_engineered_features,
)
from src.features.scaling import scale_features


DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = PROJECT_ROOT / "artifacts" / "preprocessors"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"

SUBSET = "FD001"
RUL_CAP = 125
RANDOM_STATE = 42


def main():

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AEROGUARD AI - FEATURE ENGINEERING")
    print("=" * 60)

    # 1. Load raw training data
    print("\n[1/8] Loading dataset...")

    df = load_train_data(
        data_dir=DATA_DIR,
        subset=SUBSET,
    )

    # 2. Construct RUL target
    print("\n[2/8] Constructing RUL target...")

    df = add_training_rul(
        df,
        rul_cap=RUL_CAP,
    )

    # 3. Split by engine
    print("\n[3/8] Splitting engines...")

    train_df, val_df = split_by_engine(
        df,
        validation_size=0.2,
        random_state=RANDOM_STATE,
    )

    print("Training engines:", train_df["engine_id"].nunique())
    print("Validation engines:", val_df["engine_id"].nunique())

    # 4. Sensor selection using training data only
    print("\n[4/8] Selecting sensors...")

    selected_sensors = select_variable_sensors(
        train_df
    )

    print("Selected sensors:", selected_sensors)

    # 5. Temporal feature engineering
    print("\n[5/8] Generating temporal features...")

    train_features = create_temporal_features(
        train_df,
        selected_sensors=selected_sensors,
    )

    val_features = create_temporal_features(
        val_df,
        selected_sensors=selected_sensors,
    )

    # 6. Separate features and targets
    print("\n[6/8] Separating features and targets...")

    excluded_columns = [
        "engine_id",
        "dataset",
        "RUL",
        "RUL_target",
    ]

    feature_columns = [
        col for col in train_features.columns
        if col not in excluded_columns
    ]

    X_train = train_features[feature_columns].copy()
    X_val = val_features[feature_columns].copy()

    y_train = train_features["RUL_target"].copy()
    y_val = val_features["RUL_target"].copy()

    validate_engineered_features(
        train_features,
        feature_columns,
    )

    validate_engineered_features(
        val_features,
        feature_columns,
    )

    # Ensure exact feature alignment
    X_val = X_val.reindex(columns=X_train.columns)

    # 7. Scale features for deep learning
    print("\n[7/8] Scaling features...")

    X_train_scaled, X_val_scaled, scaler = scale_features(
        X_train,
        X_val,
    )

    # 8. Save outputs
    print("\n[8/8] Saving artifacts...")

    # Unscaled datasets for tree-based models
    X_train.to_csv(
        PROCESSED_DIR / f"X_train_{SUBSET}.csv",
        index=False,
    )

    X_val.to_csv(
        PROCESSED_DIR / f"X_val_{SUBSET}.csv",
        index=False,
    )

    # Scaled datasets for LSTM
    X_train_scaled.to_csv(
        PROCESSED_DIR / f"X_train_scaled_{SUBSET}.csv",
        index=False,
    )

    X_val_scaled.to_csv(
        PROCESSED_DIR / f"X_val_scaled_{SUBSET}.csv",
        index=False,
    )

    # Targets
    y_train.to_frame("RUL_target").to_csv(
        PROCESSED_DIR / f"y_train_{SUBSET}.csv",
        index=False,
    )

    y_val.to_frame("RUL_target").to_csv(
        PROCESSED_DIR / f"y_val_{SUBSET}.csv",
        index=False,
    )

    # Metadata for later evaluation and sequence generation
    train_features[
        ["engine_id", "cycle", "RUL"]
    ].to_csv(
        PROCESSED_DIR / f"train_metadata_{SUBSET}.csv",
        index=False,
    )

    val_features[
        ["engine_id", "cycle", "RUL"]
    ].to_csv(
        PROCESSED_DIR / f"val_metadata_{SUBSET}.csv",
        index=False,
    )

    # Save preprocessing objects
    joblib.dump(
        scaler,
        MODEL_DIR / f"scaler_{SUBSET}.pkl",
    )

    joblib.dump(
        selected_sensors,
        MODEL_DIR / f"selected_sensors_{SUBSET}.pkl",
    )

    joblib.dump(
        feature_columns,
        MODEL_DIR / f"feature_columns_{SUBSET}.pkl",
    )

    # Save configuration
    metadata = {
        "dataset": SUBSET,
        "rul_cap": RUL_CAP,
        "random_state": RANDOM_STATE,
        "training_engines": int(
            train_df["engine_id"].nunique()
        ),
        "validation_engines": int(
            val_df["engine_id"].nunique()
        ),
        "selected_sensors": selected_sensors,
        "feature_count": len(feature_columns),
        "rolling_windows": [5, 10],
        "lags": [1, 3],
    }

    with open(
        REPORT_DIR / f"feature_metadata_{SUBSET}.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 60)

    print("Training feature shape:", X_train.shape)
    print("Validation feature shape:", X_val.shape)

    print("Training target shape:", y_train.shape)
    print("Validation target shape:", y_val.shape)

    print("\nArtifacts saved successfully.")


if __name__ == "__main__":
    main()
