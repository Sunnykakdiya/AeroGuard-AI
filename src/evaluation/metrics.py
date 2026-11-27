from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def nasa_score(
    y_true,
    y_pred,
) -> float:
    """
    NASA C-MAPSS asymmetric scoring function.

    Positive prediction errors (late predictions)
    receive a larger penalty.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    if not np.isfinite(y_true).all():
        raise ValueError("y_true contains invalid values.")

    if not np.isfinite(y_pred).all():
        raise ValueError("y_pred contains invalid values.")

    d = y_pred - y_true

    scores = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1.0,
        np.exp(d / 10.0) - 1.0,
    )

    return float(np.sum(scores))


def nasa_mean_score(
    y_true,
    y_pred,
) -> float:
    """Mean NASA score per observation."""
    return nasa_score(y_true, y_pred) / len(y_true)


def evaluate_regression(
    y_true,
    y_pred,
) -> dict[str, float]:
    """Calculate regression and NASA metrics."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred shapes do not match."
        )

    return {
        "MAE": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "R2": float(
            r2_score(y_true, y_pred)
        ),
        "NASA_score": float(
            nasa_score(y_true, y_pred)
        ),
        "NASA_mean_score": float(
            nasa_mean_score(y_true, y_pred)
        ),
    }
