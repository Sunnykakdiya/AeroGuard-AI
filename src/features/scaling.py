from __future__ import annotations

import pandas as pd

from sklearn.preprocessing import StandardScaler


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit scaler on training data and transform
    training and validation data.
    """

    if list(X_train.columns) != list(X_val.columns):
        raise ValueError(
            "Training and validation feature columns must match."
        )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_val_scaled = scaler.transform(X_val)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index,
    )

    X_val_scaled = pd.DataFrame(
        X_val_scaled,
        columns=X_val.columns,
        index=X_val.index,
    )

    return (
        X_train_scaled,
        X_val_scaled,
        scaler,
    )
