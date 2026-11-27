from pathlib import Path
import pandas as pd

COLUMN_NAMES = (
    ["engine_id","cycle"]
    + [f"setting_{i}"for i in range(1,4)]
    + [f"sensor_{i}"for i in range(1,22)]
)

def load_engine_data(
        file_path: str | Path,
        dataset_name: str="FD001",
) -> pd.DataFrame:
    """
    Load a NASA C-MAPSS engine dataset.

    Parameters
    ---------
    file_path:
        Path to the raw C-MAPSS txt file.
    dataset_name:
        Dataset subset, e.g. FD001.

    Returns
    ---------
    pd.DataFrame
        Parsed engine sensor data.
    """

    file_path=Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found:{file_path}"
        )

    if file_path.stat().st_size == 0:
        raise ValueError(
            f"Dataset file is empty:{file_path}"
        )

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="python"
    )

    df["dataset"] = dataset_name

    return df


def load_train_data(
        data_dir: str | Path,
        subset: str = "FD001",
) -> pd.DataFrame:
    """
    Load the run-to-failure training data.
    """
    data_dir = Path(data_dir)

    file_path = data_dir / f"train_{subset}.txt"

    return load_engine_data(
        file_path=file_path,
        dataset_name=subset,
    )


def load_test_data(
        data_dir: str | Path,
        subset: str = "FD001",
) -> pd.DataFrame:
    """
    Load the partial-life test data.
    """
    data_dir = Path(data_dir)

    file_path = data_dir / f"test_{subset}.txt"

    return load_engine_data(
        file_path=file_path,
        dataset_name=subset,
    )



def load_test_rul(
        data_dir: str | Path,
        subset: str = "FD001",
) -> pd.Series:
    """Load true RUL labels for each test engine."""
    data_dir = Path(data_dir)

    file_path = data_dir / f"RUL_{subset}.txt"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Test RUL file not found:{file_path}"
        )

    rul = pd.read_csv(
        file_path,
        sep=f"\s+",
        header=None,
        engine="python",
    ).iloc[:,0]

    return rul.astype("int64").rename("rul")