from __future__ import annotations

from sklearn.model_selection import GroupKFold


def create_engine_cv(
    n_splits: int = 5,
) -> GroupKFold:
    """
    Cross-validation that keeps engine trajectories
    separated between training and validation folds.
    """
    return GroupKFold(
        n_splits=n_splits,
    )
