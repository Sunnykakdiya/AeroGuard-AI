from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}"for i in range(1,22)]
]


def validate_schema(df:pd.DataFrame) -> None:
    """Validate that required columns exist."""
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns:{sorted(missing_columns)}"
        )


def validate_numeric_data(df:pd.DataFrame) -> None:
    """Ensure engine data columnns are numeric and finite."""
    validate_schema(df)

    numeric_columns = REQUIRED_COLUMNS

    non_numeric = [
        column
        for column in numeric_columns
        if not
        pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric:
        raise TypeError(
            f"Non-numeric columns found:{non_numeric}"
        )

    values = df[numeric_columns].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Dataset contains NaN or infinite values."
        )


def validate_engine_ids(df:pd.DataFrame) -> None:
    """Check engine ID integrity."""
    if (df["engine_id"]<=0).any():
        raise ValueError(
            "Engine IDs must be positive."
        )

    if not np.all(
        df["engine_id"] == np.floor(df["engine_id"])
    ):
        raise ValueError(
            "Engine IDs must be integers."
        )


def validate_cycles(df:pd.DataFrame) -> None:
    """Check cycle numbers and sequence continuity."""
    if (df["cycle"]<=0).any():
        raise ValueError(
            "Cycle numbers must be positive."
        )

    if not np.all(
        df["cycle"] == np.floor(df["cycle"])
    ):

        raise ValueError(
            "Cycle numbers must be integers."
        )

    duplicates = df.duplicated(
        subset=["engine_id","cycle"]
    )

    if duplicates.any():
        raise ValueError(
            "Duplicate engine-cycle records found."
        )

    sorted_df = df.sort_values(
        ["engine_id","cycle"]
    )

    cycle_diff = (
        sorted_df.groupby("engine_id")["cycle"]
        .diff()
        .dropna()
    )

    if not cycle_diff.eq(1).all():
        raise ValueError(
            "Cycle sequences are not consecutive."
        )


def validate_test_rul(
        test_df: pd.DataFrame,
        test_rul: pd.Series,
) -> None:
    """Validate endpoint RUL labels for test engines."""
    excepted_count = test_df["engine_id"].nunique()

    if len(test_rul) != excepted_count:
        raise ValueError(
            f"Expected {excepted_count} RUL values,"
            f"received {len(test_rul)}."
        )

    values = test_rul.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Test RUL contains missing or infinite values."
        )

    if (values < 0).any():
        raise ValueError(
            "Test RUL values cannot be negative."
        )


def validate_dataset(
    df: pd.DataFrame,
    dataset_type: str = "train",
) -> dict[str, Any]:
    """
    Run all structural and data-quality validations.

    Returns
    -------
    dict
        Validation summary.
    """
    validate_schema(df)
    validate_numeric_data(df)
    validate_engine_ids(df)
    validate_cycles(df)

    engine_count = df["engine_id"].nunique()
    row_count = len(df)

    cycles_per_engine = (
        df.groupby("engine_id")["cycle"]
        .agg(["min", "max", "count"])
    )

    return {
        "dataset_type": dataset_type,
        "status": "passed",
        "rows": row_count,
        "engine_count": engine_count,
        "columns": len(df.columns),
        "min_cycle": int(df["cycle"].min()),
        "max_cycle": int(df["cycle"].max()),
        "min_engine_life": int(cycles_per_engine["count"].min()),
        "max_engine_life": int(cycles_per_engine["count"].max()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_records": int(
            df.duplicated(["engine_id", "cycle"]).sum()
        ),
    }
