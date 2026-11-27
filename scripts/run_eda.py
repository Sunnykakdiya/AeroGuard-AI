from pathlib import Path
import sys

import pandas as pd

# Make project root importable when running this script directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingestion import load_train_data
from src.data.validation import validate_dataset
from src.data.quality_report import generate_quality_report
from src.data.metadata import add_training_rul, create_engine_summary
from src.visualization.eda_plots import (
    plot_engine_life_distribution,
    plot_sensor_distribution,
    plot_sensor_over_cycles,
    plot_sensor_correlation,
    plot_sensor_mean_by_cycle,
)


DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"

SUBSET = "FD001"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AEROGUARD AI - DATA INGESTION & EDA")
    print("=" * 60)

    # 1. Load dataset
    print("\n[1/6] Loading training data...")

    train_df = load_train_data(
        data_dir=DATA_DIR,
        subset=SUBSET,
    )

    print(f"Loaded shape: {train_df.shape}")

    # 2. Validate
    print("\n[2/6] Running data validation...")

    validation_summary = validate_dataset(
        train_df,
        dataset_type="train",
    )

    print(validation_summary)

    # 3. Generate quality report
    print("\n[3/6] Generating data quality report...")

    quality_report = generate_quality_report(
        train_df
    )

    quality_path = REPORT_DIR / f"quality_{SUBSET}.csv"

    quality_report.to_csv(
        quality_path,
        index=False,
    )

    print(f"Quality report saved: {quality_path}")

    # 4. Add RUL
    print("\n[4/6] Calculating training RUL...")

    train_with_rul = add_training_rul(
        train_df
    )

    # 5. Engine summary
    print("\n[5/6] Creating engine summary...")

    engine_summary = create_engine_summary(
        train_df
    )

    summary_path = (
        PROCESSED_DIR
        / f"engine_summary_{SUBSET}.csv"
    )

    engine_summary.to_csv(
        summary_path,
        index=False,
    )

    # 6. Export processed data
    processed_path = (
        PROCESSED_DIR
        / f"train_{SUBSET}_with_rul.csv"
    )

    train_with_rul.to_csv(
        processed_path,
        index=False,
    )

    print(f"Processed data saved: {processed_path}")

    # EDA
    print("\n[EDA] Creating visualizations...")

    plot_engine_life_distribution(train_df)
    plot_sensor_distribution(train_df)
    plot_sensor_over_cycles(
        train_df,
        engine_id=1,
        sensors=[
            "sensor_2",
            "sensor_3",
            "sensor_4",
        ],
    )
    plot_sensor_correlation(train_df)
    plot_sensor_mean_by_cycle(train_df)

    print("\n" + "=" * 60)
    print("PHASE 2 DATA PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
