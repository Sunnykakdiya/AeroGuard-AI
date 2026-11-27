from __future__ import annotations

import numpy as np
import pandas as pd


def create_temporal_features(
    df: pd.DataFrame,
    selected_sensors: list[str],
    windows: tuple[int, ...] = (5, 10),
    lags: tuple[int, ...] = (1, 3),
) -> pd.DataFrame:
    """
    Generate engine-wise temporal features.

    Rolling windows use current and previous observations.
    Lag features never use future observations.
    """
    if not selected_sensors:
        raise ValueError("No sensors provided.")

    required = {
        "engine_id",
        "cycle",
        *selected_sensors,
        "setting_1",
        "setting_2",
        "setting_3",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if any(w <= 0 for w in windows):
        raise ValueError("Window sizes must be positive.")

    if any(lag <= 0 for lag in lags):
        raise ValueError("Lag values must be positive.")

    result = (
        df.copy()
        .sort_values(["engine_id", "cycle"])
        .reset_index(drop=True)
    )

    grouped = result.groupby(
        "engine_id",
        sort=False,
    )

    for sensor in selected_sensors:

        # Lag features
        for lag in lags:
            lagged = grouped[sensor].shift(lag)

            result[f"{sensor}_lag_{lag}"] = (
                lagged.fillna(result[sensor])
            )

        # Rolling statistics
        for window in windows:

            rolling_mean = grouped[sensor].transform(
                lambda s: s.rolling(
                    window=window,
                    min_periods=1,
                ).mean()
            )

            rolling_std = grouped[sensor].transform(
                lambda s: s.rolling(
                    window=window,
                    min_periods=2,
                ).std(ddof=0)
            )

            result[f"{sensor}_rolling_mean_{window}"] = (
                rolling_mean
            )

            result[f"{sensor}_rolling_std_{window}"] = (
                rolling_std.fillna(0.0)
            )

        # First-order difference
        difference = grouped[sensor].diff()

        result[f"{sensor}_diff_1"] = (
            difference.fillna(0.0)
        )


    return result




def validate_engineered_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Validate generated ML feature values."""

    missing = set(feature_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing engineered features: {sorted(missing)}"
        )

    values = df[feature_columns].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Engineered features contain NaN or infinity."
        )
