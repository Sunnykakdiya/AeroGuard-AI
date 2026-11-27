from __future__ import annotations

import pandas as pd


def select_variable_sensors(
    train_df: pd.DataFrame,
    threshold: float = 1e-8,
) -> list[str]:
    """
    Select non-constant sensors using training data only.
    """
    sensor_columns = [
        col for col in train_df.columns
        if col.startswith("sensor_")
    ]

    if not sensor_columns:
        raise ValueError("No sensor columns found.")

    variances = train_df[sensor_columns].var()

    selected_sensors = variances[
        variances > threshold
    ].index.tolist()

    if not selected_sensors:
        raise ValueError(
            "No variable sensors were selected."
        )

    return selected_sensors


def get_sensor_variance(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return sensor variance for inspection."""
    sensor_columns = [
        col for col in df.columns
        if col.startswith("sensor_")
    ]

    return (
        df[sensor_columns]
        .var()
        .sort_values(ascending=False)
        .rename("variance")
        .reset_index()
        .rename(columns={"index": "sensor"})
    )
