from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_by_engine(
    df: pd.DataFrame,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split complete engine trajectories into
    training and validation datasets.
    """
    if "engine_id" not in df.columns:
        raise ValueError("engine_id column is required.")

    if not 0 < validation_size < 1:
        raise ValueError(
            "validation_size must be between 0 and 1."
        )

    groups = df["engine_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        random_state=random_state,
    )

    train_indices, val_indices = next(
        splitter.split(df, groups=groups)
    )

    train_data = df.iloc[train_indices].copy()
    val_data = df.iloc[val_indices].copy()

    train_engines = set(train_data["engine_id"])
    val_engines = set(val_data["engine_id"])

    if train_engines.intersection(val_engines):
        raise RuntimeError(
            "Engine leakage detected between splits."
        )

    return (
        train_data.sort_values(
            ["engine_id", "cycle"]
        ).reset_index(drop=True),
        val_data.sort_values(
            ["engine_id", "cycle"]
        ).reset_index(drop=True),
    )
