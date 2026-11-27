
from __future__ import annotations

import pandas as pd


def generate_quality_report(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate a column-level data quality report."""

    report = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(dtype) for dtype in df.dtypes],
        "missing_count": df.isna().sum().values,
        "missing_pct": (
            df.isna().mean().values * 100
        ),
        "unique_count": df.nunique().values,
    })

    report["is_constant"] = (
        report["unique_count"] <= 1
    )

    report["constant_warning"] = report[
        "is_constant"
    ].map(
        lambda x: "CONSTANT" if x else "OK"
    )

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    std_values = df[numeric_columns].std()

    report["std"] = report["column"].map(
        std_values.to_dict()
    )

    report["status"] = "OK"

    report.loc[
        report["missing_count"] > 0,
        "status"
    ] = "MISSING_VALUES"

    report.loc[
        report["is_constant"],
        "status"
    ] = "CONSTANT_FEATURE"

    return report.sort_values(
        ["status", "column"]
    ).reset_index(drop=True)
