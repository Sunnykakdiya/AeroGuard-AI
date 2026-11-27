
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.ingestion import COLUMN_NAMES
from src.features.engineering import (
    create_temporal_features,
    validate_engineered_features,
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
PREPROCESSOR_DIR = PROJECT_ROOT / "artifacts" / "preprocessors"

SUBSET = "FD001"
RUL_CAP = 125


# --------------------------------------------------
# LOAD SAVED ARTIFACTS
# --------------------------------------------------

def load_prediction_artifacts():

    model_path = MODEL_DIR / f"best_model_{SUBSET}.pkl"

    sensors_path = (
        PREPROCESSOR_DIR /
        f"selected_sensors_{SUBSET}.pkl"
    )

    columns_path = (
        PREPROCESSOR_DIR /
        f"feature_columns_{SUBSET}.pkl"
    )

    for path in [
        model_path,
        sensors_path,
        columns_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}"
            )

    model = joblib.load(model_path)
    selected_sensors = joblib.load(sensors_path)
    feature_columns = joblib.load(columns_path)

    if len(feature_columns) != model.n_features_in_:
        raise ValueError(
            "Feature count does not match the saved model."
        )

    return model, selected_sensors, feature_columns


# --------------------------------------------------
# VALIDATE RAW ENGINE DATA
# --------------------------------------------------

def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input engine data is empty.")

    required_columns = set(COLUMN_NAMES)

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing raw columns: {sorted(missing)}"
        )

    result = df[COLUMN_NAMES].copy()

    for column in COLUMN_NAMES:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        )

    values = result.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Raw data contains NaN or infinite values."
        )

    if (result["engine_id"] <= 0).any():
        raise ValueError("Engine IDs must be positive.")

    if (result["cycle"] <= 0).any():
        raise ValueError("Cycle numbers must be positive.")

    if result.duplicated(
        subset=["engine_id", "cycle"]
    ).any():
        raise ValueError(
            "Duplicate engine_id and cycle combinations found."
        )

    return result.sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)


# --------------------------------------------------
# MAINTENANCE STATUS
# --------------------------------------------------

def get_maintenance_status(
    predicted_rul: float,
    critical_threshold: float = 20,
    warning_threshold: float = 50,
) -> str:

    if critical_threshold >= warning_threshold:
        raise ValueError(
            "Critical threshold must be lower than warning threshold."
        )

    if predicted_rul <= critical_threshold:
        return "CRITICAL"

    elif predicted_rul <= warning_threshold:
        return "WARNING"

    return "NORMAL"


# --------------------------------------------------
# PREDICT LATEST RUL FOR EACH ENGINE
# --------------------------------------------------

def predict_latest_rul(
    df: pd.DataFrame,
) -> pd.DataFrame:

    # 1. Load saved artifacts
    model, selected_sensors, feature_columns = (
        load_prediction_artifacts()
    )

    # 2. Validate incoming raw data
    raw_data = validate_raw_data(df)

    # 3. Generate temporal features
    engineered_data = create_temporal_features(
        raw_data,
        selected_sensors=selected_sensors,
    )

    # 4. Validate all 130 features
    validate_engineered_features(
        engineered_data,
        feature_columns,
    )

    # 5. Get latest available cycle for every engine
    latest_data = (
        engineered_data
        .sort_values(["engine_id", "cycle"])
        .groupby("engine_id", sort=False)
        .tail(1)
        .copy()
    )

    # 6. Align exact training feature order
    X_latest = latest_data.reindex(
        columns=feature_columns
    )

    # 7. Predict using the existing unscaled model
    predictions = model.predict(X_latest)

    predictions = np.clip(
        predictions,
        0,
        RUL_CAP,
    )

    # 8. Prepare results
    results = pd.DataFrame({
        "engine_id": latest_data["engine_id"].to_numpy(),
        "latest_cycle": latest_data["cycle"].to_numpy(),
        "predicted_RUL": predictions,
    })

    results["maintenance_status"] = (
        results["predicted_RUL"].apply(
            get_maintenance_status
        )
    )

    return results.reset_index(drop=True)
