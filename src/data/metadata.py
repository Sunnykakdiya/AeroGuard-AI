from __future__ import annotations

import pandas as pd




def add_training_rul(
    df: pd.DataFrame,
    rul_cap: int | None = None,
) -> pd.DataFrame:
    """
    Calculate true RUL for each training observation.

    Parameters
    ----------
    df:
        Complete run-to-failure engine trajectories.
    rul_cap:
        Optional upper limit for RUL.

    Returns
    -------
    pd.DataFrame
        Original data with RUL and RUL target.
    """
    result = df.copy()

    max_cycle = (
        result.groupby("engine_id")["cycle"]
        .transform("max")
    )

    result["RUL"] = max_cycle - result["cycle"]

    if rul_cap is not None:
        if rul_cap <= 0:
            raise ValueError("RUL cap must be positive.")

        result["RUL_target"] = (
            result["RUL"].clip(upper=rul_cap)
        )
    else:
        result["RUL_target"] = result["RUL"]

    return result



def create_engine_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create one summary row per engine."""

    summary = (
        df.groupby("engine_id")
        .agg(
            first_cycle=("cycle", "min"),
            last_cycle=("cycle", "max"),
            total_cycles=("cycle", "count"),
        )
        .reset_index()
    )

    summary["engine_life_cycles"] = (
        summary["last_cycle"] - summary["first_cycle"] + 1
    )

    return summary.sort_values(
        "engine_id"
    ).reset_index(drop=True)


def attach_test_endpoint_rul(
    test_df: pd.DataFrame,
    endpoint_rul: pd.Series,
) -> pd.DataFrame:
    """
    Attach official test RUL values to the final
    observed cycle of each test engine.

    This is endpoint evaluation data, not a row-wise
    target for all test observations.
    """
    result = test_df.copy()

    engine_ids = sorted(result["engine_id"].unique())

    if len(engine_ids) != len(endpoint_rul):
        raise ValueError(
            "Test engine count does not match RUL label count."
        )

    rul_mapping = dict(
        zip(engine_ids, endpoint_rul.tolist())
    )

    result["endpoint_RUL"] = result["engine_id"].map(
        rul_mapping
    )

    result["is_final_observation"] = (
        result["cycle"]
        == result.groupby("engine_id")["cycle"].transform("max")
    )

    result["RUL_at_observation"] = (
        result.groupby("engine_id")["cycle"].transform("max")
        - result["cycle"]
        + result["endpoint_RUL"]
    )

    return result


